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
    # load follow-up latent
    fu = sample['followup_latent_path']
    fu_t = torch.as_tensor(np.load(fu)['data']).float() if isinstance(fu, str) else torch.as_tensor(fu).float()
    sample['followup_latent_path'] = fu_t
    target_shape = fu_t.shape

    latents, ages = [], []
    n_past = int(sample.get('num_past_scans', 0))
    for i in range(1, max_scans + 1):
        lp_key, age_key = f'starting{i}_latent_path', f'starting{i}_age'
        path = sample.get(lp_key)
        if i <= n_past and isinstance(path, str) and os.path.isfile(path):
            arr = np.load(path)['data']; t = torch.from_numpy(arr).float()
        else:
            t = torch.zeros(target_shape)
        # pad if needed
        if t.shape != target_shape:
            diffs = [targ - cur for targ, cur in zip(target_shape, t.shape)]
            pad = (0, max(diffs[3],0), 0, max(diffs[2],0), 0, max(diffs[1],0))
            t = F.pad(t, pad)
        latents.append(t)
        age = sample.get(age_key, 0.0)
        ages.append(float(age) if isinstance(age, (int, float)) else 0.0)
        # drop raw keys
        sample.pop(lp_key, None); sample.pop(age_key, None)
    sample['starting_latents'] = torch.stack(latents, dim=0)
    sample['starting_ages'] = torch.tensor(ages)
    return sample


def dynamic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # determine the max scans present in this minibatch
    max_scans_batch = max(int(s.get('num_past_scans', 0)) for s in batch)
    # process each sample: stack its scans
    processed = [load_and_stack_multiscans(s.copy(), max_scans_batch) for s in batch]
    # then apply default_collate on tensor fields
    collated = {}
    # find keys to collate
    keys = set(processed[0].keys())
    for k in keys:
        vals = [s[k] for s in processed]
        if isinstance(vals[0], torch.Tensor) or isinstance(vals[0], (int, float)):
            collated[k] = default_collate(vals)
        else:
            collated[k] = vals
    return collated

# torch.autograd.set_detect_anomaly(True)

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
    parser.add_argument('--lr',          default=2.5e-5,  type=float)#2.5e-5
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_csv)
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = Compose([
        LoadImageD(keys=['followup_latent_path'], reader=npz_reader),
        EnsureChannelFirstD(keys=['followup_latent_path'], channel_dim=0),
        DivisiblePadD(keys=['followup_latent_path'], k=4, mode='constant'),
        Lambda(concat_covariates),
    ])

    train_df = df[df.split == 'train']
    valid_df = df[df.split == 'valid']

    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, collate_fn=dynamic_collate_fn
    )
    valid_loader = DataLoader(
        validset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True, collate_fn=dynamic_collate_fn
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

            # context + followup
            context    = batch['context'].to(DEVICE).float()
            follow_z   = batch['followup_latent_path'].to(DEVICE) * scale_factor
            noise      = torch.randn_like(follow_z)
            timesteps  = torch.randint(
                0, scheduler.num_train_timesteps,
                (follow_z.size(0),), device=DEVICE
            ).long()
            #print(f"follow_z.shape: {follow_z.shape}, context.shape: {context.shape}")

            # ** now multi‐scan **
            zs_all = batch['starting_latents'].to(DEVICE) * scale_factor
            a_all  = batch['starting_ages'].to(DEVICE)
            B, N, C, H, W, D = zs_all.shape
            #print(f"zs_all.shape: {zs_all.shape}, a_all.shape: {a_all.shape}")
            # print(a_all)

            # list of [B,C,H,W,D] and [B]
            z_list = list(torch.unbind(zs_all, dim=1))
            a_list = list(torch.unbind(a_all,  dim=1))

            conds = []
            for z_i, a_i in zip(z_list, a_list):
                # if z_i.dim() == 5:
                #     # z_i = z_i.unsqueeze(1)
                #     print("yes it is 5")
                # age_map = a_i.view(B,1,*([1]*(z_i.dim()-2))).expand(B,1,*z_i.shape[-3:])
                # conds.append(torch.cat([z_i, age_map], dim=1))

                concatenating_age      = a_i.view(B, 1, 1, 1, 1).expand(B, 1, *z_i.shape[-3:])
                controlnet_condition   = torch.cat([ z_i, concatenating_age ], dim=1)
                conds.append(controlnet_condition)

            #print(f"conds shape at index 0 : {conds[0].shape}")

            noised = scheduler.add_noise(follow_z, noise=noise, timesteps=timesteps)
            #print(f"noised tensor:{noised.shape}")

            with torch.set_grad_enabled(is_train):
                with autocast():
                    pred = None
                    for j, c in enumerate(conds):
                        dh, mh = controlnet(
                            x=noised.float(),
                            timesteps=timesteps,
                            context=context,
                            controlnet_cond=c.float()
                        )
                        if torch.isnan(c).any() or torch.isinf(c).any():
                            print(f"NaN or inf in controlnet_cond at step {step}, skipping batch")
                            continue
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
                            # your original “delta” update (no in-place operations)
                            upd = pred.clone()
                            norm_up = upd.view(-1).norm()
                            delta   = (pred - prev).abs()
                            delta   = delta / (delta.view(-1).norm() + 1e-4)
                            age_diff = (a_list[j] - a_list[j-1])\
                                .view(B, *([1]*(upd.dim()-1))).float()
                            age_diff = torch.clamp(age_diff, min=1.0)
                            upd = upd + ((delta * pred)/(age_diff+1e-4))*(j/(j+1))
                            upd = (upd/(upd.view(-1).norm()+1e-4))*norm_up
                            prev = upd  # no in-place assignment here
                            # print("delta_norm:", delta_norm.item(), "upd_norm:", upd_norm.item(), "norm_up:", norm_up.item())

                        # if(j == 0):
                        #     prev_noise_step = noise_pred
                        # else:
                        #     updated_curr_tensor = noise_pred.clone()
                        #     updated_curr_tensor_norm = updated_curr_tensor.view(-1).norm()
                        #     temp_latent = torch.abs(noise_pred - prev_noise_step)
                        #     temp_latent = temp_latent / (temp_latent.view(-1).norm() + 1e-8)  # norm over all dims

                        #     updated_curr_tensor += (((temp_latent * noise_pred) / ((a_list[j] - a_list[j-1])*1.0))*(j/(j+1)))
                        #     updated_curr_tensor = (updated_curr_tensor  / (updated_curr_tensor.view(-1).norm() + 1e-8)) * updated_curr_tensor_norm
                        #     prev_noise_step = updated_curr_tensor
                            
                    loss = F.mse_loss(pred.float(), noise.float())
                    #print(f"loss: {loss.item()}")
                if is_train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                writer.add_scalar(f'{mode}/batch-mse', loss.item(), steps[mode])
                epoch_loss += loss.item()
                steps[mode] += 1
                pbar.set_postfix(loss=epoch_loss/(step+1))

            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss/len(loader), epoch)
            # images_to_tensorboard(writer, epoch, mode,
            #                       autoencoder, diffusion, controlnet,
            #                       trainset if mode=='train' else validset,
            #                       scale_factor)

        # checkpoint
        if epoch > 2:
            torch.save(controlnet.state_dict(),
                       os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth'))
