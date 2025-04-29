import os
import argparse
import warnings

from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai import transforms
from monai.transforms import (
    LoadImageD, EnsureChannelFirstD, DivisiblePadD, Lambda, Compose
)
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

from brlp import const, utils, networks
from brlp import get_dataset_from_pd, sample_using_controlnet_and_z

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(sample):
    # unchanged
    conds = [
        sample['followup_age'],
        sample['sex'],
        sample['followup_diagnosis'],
        sample['followup_cerebral_cortex'],
        sample['followup_hippocampus'],
        sample['followup_amygdala'],
        sample['followup_cerebral_white_matter'],
        sample['followup_lateral_ventricle'],
    ]
    sample['context'] = torch.tensor(conds).unsqueeze(0)
    return sample


def load_and_stack_multiscans(sample: dict, max_scans: int) -> dict:
    # 1) Turn follow-up into a pure torch.Tensor (MetaTensor or np.ndarray)
    fu = sample['followup_latent_path']
    if isinstance(fu, np.ndarray):
        fu_t = torch.from_numpy(fu).float()
    else:
        fu_t = torch.as_tensor(fu).float()
    sample['followup_latent_path'] = fu_t

    # target shape = (C, H, W, D)
    target_shape = fu_t.shape  
    n_past       = int(sample.get('num_past_scans', 0))

    latents = []
    ages    = []

    for i in range(1, max_scans + 1):
        lp_key  = f'starting{i}_latent_path'
        age_key = f'starting{i}_age'
        path    = sample.get(lp_key, None)

        # load or zero-fill
        if i <= n_past and isinstance(path, str) and os.path.isfile(path):
            arr = np.load(path)['data']
            t   = torch.from_numpy(arr).float()
        else:
            t   = torch.zeros(target_shape, dtype=fu_t.dtype)

        # **pad** t to target_shape if needed
        if t.shape != target_shape:
            c1,h1,w1,d1 = t.shape
            _,h0,w0,d0 = target_shape
            # compute how much to pad on each spatial dim
            dh, dw, dd = h0-h1, w0-w1, d0-d1
            # pad format: (pad_left_D, pad_right_D,
            #              pad_left_W, pad_right_W,
            #              pad_left_H, pad_right_H)
            pad = (0, max(0, dd),  0, max(0, dw),  0, max(0, dh))
            t = F.pad(t, pad)

        latents.append(t)

        # age or zero
        age = sample.get(age_key, None)
        ages.append(float(age) if isinstance(age, (int, float)) else 0.0)

        # remove raw keys
        sample.pop(lp_key,  None)
        sample.pop(age_key, None)

    # stack into fixed-length tensors
    sample['starting_latents'] = torch.stack(latents, dim=0)            # (max_scans, C, H, W, D)
    sample['starting_ages']    = torch.tensor(ages, dtype=torch.float32)  # (max_scans,)
    return sample

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # find keys common to every sample
    common_keys = set(batch[0].keys())
    for samp in batch[1:]:
        common_keys &= set(samp.keys())

    collated: Dict[str, Any] = {}
    for key in common_keys:
        vals = [samp[key] for samp in batch]
        first = vals[0]
        if isinstance(first, (int, float, torch.Tensor)):
            collated[key] = default_collate(vals)
        else:
            collated[key] = vals  # list of strings or other metadata

    return collated


def images_to_tensorboard(writer, epoch, mode,
                          autoencoder, diffusion, controlnet,
                          dataset, scale_factor):
    resample = transforms.Spacing(pixdim=1.5)
    idxs = np.random.choice(len(dataset), 3, replace=False)
    for ti, i in enumerate(idxs):
        s = dataset[i]
        # first scan only for viz
        z0 = s['starting_latents'][0] * scale_factor
        a0 = s['starting_ages'][0]
        ctx = s['context']

        # load T1 images
        SI = torch.from_numpy(nib.load(s['starting_image_path']).get_fdata()).unsqueeze(0)
        FI = torch.from_numpy(nib.load(s['followup_image_path']).get_fdata()).unsqueeze(0)
        SI = resample(SI).squeeze(0)
        FI = resample(FI).squeeze(0)

        PI = sample_using_controlnet_and_z(
            autoencoder, diffusion, controlnet,
            starting_z=z0, starting_a=a0,
            context=ctx, device=DEVICE,
            scale_factor=scale_factor
        )
        utils.tb_display_cond_generation(
            writer, epoch, f'{mode}/cmp_{ti}',
            starting_image=SI, followup_image=FI, predicted_image=PI
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--cache_dir',   required=True, type=str)
    parser.add_argument('--output_dir',  required=True, type=str)
    parser.add_argument('--aekl_ckpt',   required=True, type=str)
    parser.add_argument('--diff_ckpt',   required=True, type=str)
    parser.add_argument('--cnet_ckpt',   default=None,    type=str)
    parser.add_argument('--num_workers', default=8,       type=int)
    parser.add_argument('--n_epochs',    default=5,       type=int)
    parser.add_argument('--batch_size',  default=8,       type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    # any column like "starting17_latent_path" → extract the "17"
    start_cols = [c for c in df.columns if c.startswith('starting') and c.endswith('_latent_path')]
    MAX_SCANS = max(int(c[len('starting'):c.find('_latent_path')]) for c in start_cols)

    # we only let MONAI load & pad the *follow‐up* latent:
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        LoadImageD(keys=['followup_latent_path'], reader=npz_reader),
        EnsureChannelFirstD(keys=['followup_latent_path'], channel_dim=0),
        DivisiblePadD(keys=['followup_latent_path'], k=4, mode='constant'),
        # pass MAX_SCANS into our lambda
        Lambda(lambda sample, ms=MAX_SCANS: load_and_stack_multiscans(sample, ms)),
        Lambda(func=concat_covariates),
    ])


    train_df = df[df.split == 'train']
    valid_df = df[df.split == 'valid']

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True
    )
    valid_loader = DataLoader(
        validset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True
    )

    # — model init (unchanged) —
    autoencoder = networks.init_autoencoder(args.aekl_ckpt).to(DEVICE)
    diffusion   = networks.init_latent_diffusion(args.diff_ckpt).to(DEVICE)
    controlnet  = networks.init_controlnet().to(DEVICE)

    if args.cnet_ckpt:
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)
    for p in diffusion.parameters():
        p.requires_grad = False

    scaler    = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    # compute scale
    with torch.no_grad(), autocast():
        z0 = trainset[0]['followup_latent_path']
    scale_factor = 1.0 / torch.std(z0)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205
    )

    writer = SummaryWriter()
    loaders = {'train': train_loader, 'valid': valid_loader}
    steps   = {'train': 0, 'valid': 0}

    for epoch in range(args.n_epochs):
        for mode, loader in loaders.items():
            is_train = (mode == 'train')
            controlnet.train() if is_train else controlnet.eval()

            epoch_loss = 0.0
            pbar = tqdm(enumerate(loader), total=len(loader))
            for step, batch in pbar:
                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                
                # for k, v in batch.items():
                #     try:
                #         print(k, v.shape)
                #     except:
                #         print(k, len(v))
                # context + followup
                context    = batch['context'].to(DEVICE).float()
                follow_z   = batch['followup_latent_path'].to(DEVICE) * scale_factor
                noise      = torch.randn_like(follow_z)
                timesteps  = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (follow_z.size(0),), device=DEVICE
                ).long()

                # ** now multi‐scan **
                zs_all = batch['starting_latents'].to(DEVICE) * scale_factor
                a_all  = batch['starting_ages'].to(DEVICE)
                B, N, C, H, W, D = zs_all.shape

                # list of [B,C,H,W,D] and [B]
                z_list = list(torch.unbind(zs_all, dim=1))
                a_list = list(torch.unbind(a_all,  dim=1))

                conds = []
                for z_i, a_i in zip(z_list, a_list):
                    if z_i.dim() == 4:
                        z_i = z_i.unsqueeze(1)
                    age_map = a_i.view(B,1,*([1]*(z_i.dim()-2))).expand(B,1,*z_i.shape[-3:])
                    conds.append(torch.cat([z_i, age_map], dim=1))

                noised = scheduler.add_noise(follow_z, noise=noise, timesteps=timesteps)

                with torch.set_grad_enabled(is_train):
                    with autocast():
                        prev = None
                        for j, c in enumerate(conds):
                            dh, mh = controlnet(
                                x=noised.float(),
                                timesteps=timesteps,
                                context=context,
                                controlnet_cond=c.float()
                            )
                            pred = diffusion(
                                x=noised.float(),
                                timesteps=timesteps,
                                context=context,
                                down_block_additional_residuals=dh,
                                mid_block_additional_residual=mh
                            )
                            if j == 0:
                                prev = pred
                            else:
                                # your original “delta” update
                                upd = pred.clone()
                                norm_up = upd.view(-1).norm()
                                delta   = (pred - prev).abs()
                                delta   = delta / (delta.view(-1).norm() + 1e-8)
                                age_diff = (a_list[j] - a_list[j-1])\
                                    .view(B, *([1]*(upd.dim()-1))).float()
                                upd   += ((delta * pred)/(age_diff+1e-8))*(j/(j+1))
                                upd    = (upd/(upd.view(-1).norm()+1e-8))*norm_up
                                prev   = upd
                        loss = F.mse_loss(prev.float(), noise.float())

                if is_train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-mse', loss.item(), steps[mode])
                epoch_loss += loss.item()
                steps[mode] += 1
                pbar.set_postfix(loss=epoch_loss/(step+1))

            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss/len(loader), epoch)
            images_to_tensorboard(writer, epoch, mode,
                                  autoencoder, diffusion, controlnet,
                                  trainset if mode=='train' else validset,
                                  scale_factor)

        # checkpoint
        if epoch > 2:
            torch.save(controlnet.state_dict(),
                       os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth'))
