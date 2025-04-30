# import os
# import argparse
# import warnings

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
# import nibabel as nib
# from torch.utils.data import DataLoader
# from torch.cuda.amp import autocast, GradScaler
# from torch.utils.tensorboard import SummaryWriter
# from monai import transforms
# from monai.data.image_reader import NumpyReader
# from generative.networks.schedulers import DDPMScheduler
# from tqdm import tqdm
# from torch.utils.data._utils.collate import default_collate

# from brlp import const
# from brlp import utils
# from brlp import networks
# from brlp import (
#     get_dataset_from_pd, 
#     sample_using_controlnet_and_z
# )


# warnings.filterwarnings("ignore")
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# def concat_covariates(_dict):
#     """
#     Provide context for cross-attention layers and concatenate the
#     covariates in the channel dimension.
#     """
#     conditions = [
#         _dict['followup_age'], 
#         _dict['sex'], 
#         _dict['followup_diagnosis'], 
#         _dict['followup_cerebral_cortex'], 
#         _dict['followup_hippocampus'], 
#         _dict['followup_amygdala'], 
#         _dict['followup_cerebral_white_matter'], 
#         _dict['followup_lateral_ventricle']        
#     ]
#     _dict['context'] = torch.tensor(conditions).unsqueeze(0)
#     return _dict


# def images_to_tensorboard(
#     writer,
#     epoch, 
#     mode, 
#     autoencoder, 
#     diffusion, 
#     controlnet, 
#     dataset,
#     scale_factor
# ):
#     """
#     Visualize the generation on tensorboard
#     """
#     resample_fn = transforms.Spacing(pixdim=1.5)
#     random_indices = np.random.choice( range(len(dataset)), 3 ) 

#     for tag_i, i in enumerate(random_indices):

#         starting_z = dataset[i]['starting_latent'] * scale_factor
#         context    = dataset[i]['context']
#         starting_a = dataset[i]['starting_age']

#         starting_image = torch.from_numpy(nib.load(dataset[i]['starting_image']).get_fdata()).unsqueeze(0)
#         followup_image = torch.from_numpy(nib.load(dataset[i]['followup_image']).get_fdata()).unsqueeze(0)
#         starting_image = resample_fn(starting_image).squeeze(0)
#         followup_image = resample_fn(followup_image).squeeze(0)

#         predicted_image = sample_using_controlnet_and_z(
#             autoencoder=autoencoder, 
#             diffusion=diffusion, 
#             controlnet=controlnet, 
#             starting_z=starting_z, 
#             starting_a=starting_a, 
#             context=context, 
#             device=DEVICE,
#             scale_factor=scale_factor
#         )

#         utils.tb_display_cond_generation(
#             writer=writer, 
#             step=epoch, 
#             tag=f'{mode}/comparison_{tag_i}',
#             starting_image=starting_image, 
#             followup_image=followup_image, 
#             predicted_image=predicted_image
#         )

# def custom_collate_fn(batch):
#     numeric_batch = []
#     for item in batch:
#         filtered = {k: v for k, v in item.items() if isinstance(v, (int, float, torch.Tensor))}
#         numeric_batch.append(filtered)
#     return default_collate(numeric_batch)


# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset_csv', required=True,   type=str)
#     parser.add_argument('--cache_dir',   required=True,   type=str)
#     parser.add_argument('--output_dir',  required=True,   type=str)
#     parser.add_argument('--aekl_ckpt',   required=True,   type=str)
#     parser.add_argument('--diff_ckpt',   required=True,   type=str)
#     parser.add_argument('--cnet_ckpt',   default=None,    type=str)
#     parser.add_argument('--num_workers', default=8,       type=int)
#     parser.add_argument('--n_epochs',    default=5,       type=int)
#     parser.add_argument('--batch_size',  default=16,      type=int)
#     parser.add_argument('--lr',          default=2.5e-5,  type=float)
    
#     args = parser.parse_args()


#     npz_reader = NumpyReader(npz_keys=['data'])
#     transforms_fn = transforms.Compose([
#         transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz_reader), 
#         transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0), 
#         transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'), 
#         transforms.Lambda(func=concat_covariates),
#     ])

#     dataset_df = pd.read_csv(args.dataset_csv)
#     train_df = dataset_df[ dataset_df.split == 'train' ]
#     valid_df = dataset_df[ dataset_df.split == 'valid' ]
#     trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
#     validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

#     # print(trainset[0])

#     train_loader = DataLoader(dataset=trainset, 
#                               num_workers=args.num_workers, 
#                               batch_size=args.batch_size, 
#                               shuffle=True, 
#                               persistent_workers=True, 
#                               pin_memory=True,
#                               collate_fn=custom_collate_fn)

#     valid_loader = DataLoader(dataset=validset, 
#                               num_workers=args.num_workers, 
#                               batch_size=args.batch_size, 
#                               shuffle=True, 
#                               persistent_workers=True, 
#                               pin_memory=True,
#                               collate_fn=custom_collate_fn)


#     # eorbfow

#     autoencoder = networks.init_autoencoder(args.aekl_ckpt)
#     diffusion   = networks.init_latent_diffusion(args.diff_ckpt)
#     controlnet  = networks.init_controlnet()

#     if args.cnet_ckpt is not None:
#         print('Resuming training...')
#         controlnet.load_state_dict(torch.load(args.cnet_ckpt))
#     else:
#         print('Copying weights from diffusion model')
#         controlnet.load_state_dict(diffusion.state_dict(), strict=False)

#     # freeze the unet weights
#     for p in diffusion.parameters():
#         p.requires_grad = False

#     # Move everything to DEVICE
#     autoencoder.to(DEVICE)
#     diffusion.to(DEVICE)
#     controlnet.to(DEVICE)

#     scaler = GradScaler()
#     optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

#     with torch.no_grad():
#         with autocast(enabled=True):
#             z = trainset[0]['followup_latent']

#     scale_factor = 1 / torch.std(z)
#     print(f"Scaling factor set to {scale_factor}")

#     scheduler = DDPMScheduler(num_train_timesteps=1000, 
#                               schedule='scaled_linear_beta', 
#                               beta_start=0.0015, 
#                               beta_end=0.0205)
    
#     writer = SummaryWriter()

#     global_counter  = { 'train': 0, 'valid': 0 }
#     loaders         = { 'train': train_loader, 'valid': valid_loader }
#     datasets        = { 'train': trainset, 'valid': validset }


#     for epoch in range(args.n_epochs):
        
#         for mode in loaders.keys():
#             print('mode:', mode)
#             loader = loaders[mode]
#             controlnet.train() if mode == 'train' else controlnet.eval()
#             epoch_loss = 0.
#             progress_bar = tqdm(enumerate(loader), total=len(loader))
#             progress_bar.set_description(f"Epoch {epoch}")

#             for step, batch in progress_bar:
#                 # print(batch['num_past_scans'])
#                 print(batch['context'].shape)
                
#                 if mode == 'train':
#                     optimizer.zero_grad(set_to_none=True)

#                 with torch.set_grad_enabled(mode == 'train'):
                    

#                     starting_z = batch['starting_latent'].to(DEVICE)  * scale_factor
#                     followup_z = batch['followup_latent'].to(DEVICE)  * scale_factor
#                     # starting_z = followup_z.clone()
#                     context    = batch['context'].to(DEVICE)
#                     starting_a = batch['starting_age'].to(DEVICE)
#                     print(f"starting_z.shape: {starting_z.shape}, starting_a.shape: {starting_a.shape}")
#                     print(f"followup_z.shape: {followup_z.shape}, context.shape: {context.shape}")

#                     n = starting_z.shape[0] 

#                     with autocast(enabled=True):

#                         concatenating_age      = starting_a.view(n, 1, 1, 1, 1).expand(n, 1, *starting_z.shape[-3:])
#                         controlnet_condition   = torch.cat([ starting_z, concatenating_age ], dim=1)

#                         print(f"controlnet_condition.shape: {controlnet_condition.shape}")

#                         noise = torch.randn_like(followup_z).to(DEVICE)
#                         timesteps = torch.randint(0, scheduler.num_train_timesteps, (n,), device=DEVICE).long()
#                         images_noised = scheduler.add_noise(followup_z, noise=noise, timesteps=timesteps)

#                         down_h, mid_h = controlnet(
#                             x=images_noised.float(), 
#                             timesteps=timesteps, 
#                             context=context.float(),
#                             controlnet_cond=controlnet_condition.float()
#                         )

#                         noise_pred = diffusion(
#                             x=images_noised.float(), 
#                             timesteps=timesteps, 
#                             context=context.float(), 
#                             down_block_additional_residuals=down_h,
#                             mid_block_additional_residual=mid_h
#                         )

#                         loss = F.mse_loss(noise_pred.float(), noise.float())

#                 if mode == 'train':
#                     scaler.scale(loss).backward()
#                     scaler.step(optimizer)
#                     scaler.update()
                
#                 #-------------------------------
#                 # Iteration end
#                 writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
#                 epoch_loss += loss.item()
#                 progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
#                 global_counter[mode] += 1

#             # Epoch loss
#             epoch_loss = epoch_loss / len(loader)
#             writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)
            
#             # Logging visualization
#             # images_to_tensorboard(
#             #     writer=writer,
#             #     epoch=epoch,
#             #     mode=mode, 
#             #     autoencoder=autoencoder, 
#             #     diffusion=diffusion, 
#             #     controlnet=controlnet,
#             #     dataset=datasets[mode], 
#             #     scale_factor=scale_factor
#             # )

#         if epoch > 2:
#             savepath = os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth')
#             torch.save(controlnet.state_dict(), savepath)

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
    parser.add_argument('--batch_size',  default=4,       type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)#2.5e-5
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
