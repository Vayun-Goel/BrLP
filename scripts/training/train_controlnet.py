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
    LoadImageD, EnsureChannelFirstD, DivisiblePadD, Lambda
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


def load_and_stack_multiscans(sample: dict) -> dict:
    """
    Load follow‐up latent into a tensor, then for each of the
    sample['num_past_scans'] past scans:
      - load .npz → tensor
      - pad spatial dims to match follow‐up
      - collect into a Python list
    Store:
      sample['starting_latents'] : List[Tensor{C,H,W,D}]
      sample['starting_ages']    : List[float]
    """
    # --- follow‐up latent → tensor
    fu = sample['followup_latent_path']
    # MONAI gives us a dict {'data': array} after LoadImageD, so handle both
    if isinstance(fu, dict) and 'data' in fu:
        fu_t = torch.from_numpy(fu['data']).float()
    else:
        fu_t = torch.as_tensor(fu).float()
    sample['followup_latent_path'] = fu_t

    target_shape = fu_t.shape  # (C, H, W, D)
    n_past = int(sample.get('num_past_scans', 0))

    latents: List[torch.Tensor] = []
    ages:    List[float]        = []

    for i in range(1, n_past + 1):
        lp_key = f'starting{i}_latent_path'
        age_key = f'starting{i}_age'
        path = sample.get(lp_key, None)

        if isinstance(path, str) and os.path.isfile(path):
            arr = np.load(path)['data']
            t   = torch.from_numpy(arr).float()
        else:
            t   = torch.zeros(target_shape, dtype=fu_t.dtype)

        # pad to match follow‐up spatial dims
        if t.shape != target_shape:
            _, h0, w0, d0 = target_shape
            _, h1, w1, d1 = t.shape
            pad = (0, max(0, d0-d1),
                   0, max(0, w0-w1),
                   0, max(0, h0-h1))
            t = F.pad(t, pad)

        latents.append(t)
        age = sample.get(age_key)
        ages.append(float(age) if isinstance(age, (int, float)) else 0.0)

        # drop the raw placeholders
        sample.pop(lp_key, None)
        sample.pop(age_key, None)

    sample['starting_latents'] = latents
    sample['starting_ages']    = ages
    return sample


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # --- 1) basic collate for everything except the variable‐length fields ---
    common = set(batch[0].keys())
    for s in batch[1:]:
        common &= set(s.keys())

    out: Dict[str, Any] = {}
    for k in common:
        if k in ('starting_latents', 'starting_ages'):
            continue
        vals = [s[k] for s in batch]
        if isinstance(vals[0], (int, float, torch.Tensor)):
            out[k] = default_collate(vals)
        else:
            out[k] = vals

    # --- 2) build per‐sample lists of latents & ages ---
    lat_lists = []
    age_lists = []
    for s in batch:
        # extract and normalize to Python lists
        raw_l = s['starting_latents']
        if isinstance(raw_l, torch.Tensor):
            # shape (N, C, H, W, D) → list of N tensors
            lats = list(torch.unbind(raw_l, dim=0))
        else:
            lats = raw_l

        raw_a = s['starting_ages']
        if isinstance(raw_a, torch.Tensor):
            ages = raw_a.tolist()
        else:
            ages = raw_a

        lat_lists.append(lats)
        age_lists.append(ages)

    # --- 3) find batch‐max and pad each sample up to that ---
    batch_max = max(len(l) for l in lat_lists)
    # infer C,H,W,D from follow‐up (all samples share shape)
    C, H, W, D = batch[0]['followup_latent_path'].shape

    padded_lats = []
    padded_ages = []
    for lats, ages in zip(lat_lists, age_lists):
        n = len(lats)
        if n < batch_max:
            zeros = [torch.zeros((C, H, W, D), dtype=lats[0].dtype)
                     for _ in range(batch_max - n)]
            lats = lats + zeros
            ages = ages + [0.0] * (batch_max - n)
        padded_lats.append(torch.stack(lats, dim=0))
        padded_ages.append(torch.tensor(ages, dtype=torch.float32))

    # --- 4) stack into batch tensors ---
    out['starting_latents'] = torch.stack(padded_lats, dim=0)  # (B, batch_max, C, H, W, D)
    out['starting_ages']    = torch.stack(padded_ages,    dim=0)  # (B, batch_max)

    return out


def images_to_tensorboard(writer, epoch, mode,
                          autoencoder, diffusion, controlnet,
                          dataset, scale_factor):
    resample = transforms.Spacing(pixdim=1.5)
    indices = np.random.choice(len(dataset), 3, replace=False)
    for ti, idx in enumerate(indices):
        s = dataset[idx]
        z0 = s['starting_latents'][0] * scale_factor
        a0 = s['starting_ages'][0]
        ctx = s['context']

        SI = torch.from_numpy(nib.load(s['starting_image_path']).get_fdata()).unsqueeze(0)
        FI = torch.from_numpy(nib.load(s['followup_image_path']).get_fdata()).unsqueeze(0)
        SI, FI = resample(SI).squeeze(0), resample(FI).squeeze(0)

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
    parser.add_argument('--dataset_csv', required=True)
    parser.add_argument('--cache_dir',   required=True)
    parser.add_argument('--output_dir',  required=True)
    parser.add_argument('--aekl_ckpt',   required=True)
    parser.add_argument('--diff_ckpt',   required=True)
    parser.add_argument('--cnet_ckpt',   default=None)
    parser.add_argument('--num_workers', default=8,  type=int)
    parser.add_argument('--n_epochs',    default=5,  type=int)
    parser.add_argument('--batch_size',  default=8,  type=int)
    parser.add_argument('--lr',          default=2.5e-5, type=float)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)

    # MONAI only loads & pads the *follow‐up* latent;
    # our lambda handles loading & stacking past scans by num_past_scans.
    reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        LoadImageD(keys=['followup_latent_path'], reader=reader),
        EnsureChannelFirstD(keys=['followup_latent_path'], channel_dim=0),
        DivisiblePadD(keys=['followup_latent_path'], k=4, mode='constant'),
        Lambda(func=load_and_stack_multiscans),
        Lambda(func=concat_covariates),
    ])

    train_df = df[df.split == 'train']
    valid_df = df[df.split == 'valid']

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    valid_loader = DataLoader(
        validset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )

    # model init
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

    # compute scale from first follow‐up
    with torch.no_grad(), autocast():
        z0 = trainset[0]['followup_latent_path']
    scale_factor = 1.0 / torch.std(z0)

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule='scaled_linear_beta',
        beta_start=0.0015, beta_end=0.0205
    )

    writer  = SummaryWriter()
    loaders = {'train': train_loader, 'valid': valid_loader}
    steps   = {'train': 0, 'valid': 0}

    for epoch in range(args.n_epochs):
        for mode, loader in loaders.items():
            is_train = (mode == 'train')
            controlnet.train() if is_train else controlnet.eval()

            epoch_loss = 0.0
            pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{mode} Epoch {epoch}")
            for step, batch in pbar:
                if is_train:
                    optimizer.zero_grad(set_to_none=True)

                context  = batch['context'].to(DEVICE).float()
                follow_z = batch['followup_latent_path'].to(DEVICE) * scale_factor
                noise    = torch.randn_like(follow_z)
                timesteps = torch.randint(
                    0, scheduler.num_train_timesteps,
                    (follow_z.size(0),), device=DEVICE
                ).long()

                zs_all = batch['starting_latents'].to(DEVICE) * scale_factor
                print(zs_all.shape)
                a_all  = batch['starting_ages'].to(DEVICE)
                B, N, C, H, W, D = zs_all.shape

                z_list = list(torch.unbind(zs_all, dim=1))
                a_list = list(torch.unbind(a_all,  dim=1))

                conds = []
                for z_i, a_i in zip(z_list, a_list):
                    if z_i.dim() == 4:
                        z_i = z_i.unsqueeze(1)
                    age_map = a_i.view(B,1,*([1]*(z_i.dim()-2))).expand(B,1,*z_i.shape[-3:])
                    conds.append(torch.cat([z_i, age_map], dim=1))

                noised = scheduler.add_noise(follow_z, noise=noise, timesteps=timesteps)

                with torch.set_grad_enabled(is_train), autocast():
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
                            upd = pred.clone()
                            norm = upd.view(-1).norm()
                            delta = (pred - prev).abs()
                            delta = delta / (delta.view(-1).norm() + 1e-8)
                            age_diff = (a_list[j] - a_list[j-1])\
                                .view(B, *([1]*(upd.dim()-1))).float()
                            upd = upd + ((delta * pred)/(age_diff+1e-8))*(j/(j+1))
                            prev = upd * (norm / (upd.view(-1).norm()+1e-8))

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

        # save checkpoint after epoch 2
        if epoch > 2:
            torch.save(controlnet.state_dict(),
                       os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth'))
