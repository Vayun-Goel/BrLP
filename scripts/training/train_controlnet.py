import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import nibabel as nib
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from monai import transforms
from monai.data.image_reader import NumpyReader
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate

from brlp import const
from brlp import utils
from brlp import networks
from brlp import (
    get_dataset_from_pd, 
    sample_using_controlnet_and_z
)


warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def concat_covariates(_dict):
    """
    Provide context for cross-attention layers and concatenate the
    covariates in the channel dimension.
    """
    conditions = [
        _dict['followup_age'], 
        _dict['sex'], 
        _dict['followup_diagnosis'], 
        _dict['followup_cerebral_cortex'], 
        _dict['followup_hippocampus'], 
        _dict['followup_amygdala'], 
        _dict['followup_cerebral_white_matter'], 
        _dict['followup_lateral_ventricle']        
    ]
    _dict['context'] = torch.tensor(conditions).unsqueeze(0)
    return _dict


def images_to_tensorboard(
    writer,
    epoch, 
    mode, 
    autoencoder, 
    diffusion, 
    controlnet, 
    dataset,
    scale_factor
):
    """
    Visualize the generation on tensorboard
    """
    resample_fn = transforms.Spacing(pixdim=1.5)
    random_indices = np.random.choice( range(len(dataset)), 3 ) 

    for tag_i, i in enumerate(random_indices):

        starting_z = dataset[i]['starting_latent'] * scale_factor
        context    = dataset[i]['context']
        starting_a = dataset[i]['starting_age']

        starting_image = torch.from_numpy(nib.load(dataset[i]['starting_image']).get_fdata()).unsqueeze(0)
        followup_image = torch.from_numpy(nib.load(dataset[i]['followup_image']).get_fdata()).unsqueeze(0)
        starting_image = resample_fn(starting_image).squeeze(0)
        followup_image = resample_fn(followup_image).squeeze(0)

        predicted_image = sample_using_controlnet_and_z(
            autoencoder=autoencoder, 
            diffusion=diffusion, 
            controlnet=controlnet, 
            starting_z=starting_z, 
            starting_a=starting_a, 
            context=context, 
            device=DEVICE,
            scale_factor=scale_factor
        )

        utils.tb_display_cond_generation(
            writer=writer, 
            step=epoch, 
            tag=f'{mode}/comparison_{tag_i}',
            starting_image=starting_image, 
            followup_image=followup_image, 
            predicted_image=predicted_image
        )

def custom_collate_fn(batch):
    numeric_batch = []
    for item in batch:
        filtered = {k: v for k, v in item.items() if isinstance(v, (int, float, torch.Tensor))}
        numeric_batch.append(filtered)
    return default_collate(numeric_batch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True,   type=str)
    parser.add_argument('--cache_dir',   required=True,   type=str)
    parser.add_argument('--output_dir',  required=True,   type=str)
    parser.add_argument('--aekl_ckpt',   required=True,   type=str)
    parser.add_argument('--diff_ckpt',   required=True,   type=str)
    parser.add_argument('--cnet_ckpt',   default=None,    type=str)
    parser.add_argument('--num_workers', default=8,       type=int)
    parser.add_argument('--n_epochs',    default=5,       type=int)
    parser.add_argument('--batch_size',  default=16,      type=int)
    parser.add_argument('--lr',          default=2.5e-5,  type=float)
    
    args = parser.parse_args()


    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting1_latent_path', 'followup_latent_path'], reader=npz_reader), 
        transforms.EnsureChannelFirstD(keys=['starting1_latent_path', 'followup_latent_path'], channel_dim=0), 
        transforms.DivisiblePadD(keys=['starting1_latent_path', 'followup_latent_path'], k=4, mode='constant'), 
        transforms.Lambda(func=concat_covariates),
    ])

    dataset_df = pd.read_csv(args.dataset_csv)
    train_df = dataset_df[ dataset_df.split == 'train' ]
    valid_df = dataset_df[ dataset_df.split == 'valid' ]
    trainset = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    validset = get_dataset_from_pd(valid_df, transforms_fn, args.cache_dir)

    train_loader = DataLoader(dataset=trainset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True,
                              collate_fn=custom_collate_fn)

    valid_loader = DataLoader(dataset=validset, 
                              num_workers=args.num_workers, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              persistent_workers=True, 
                              pin_memory=True,
                              collate_fn=custom_collate_fn)


    # eorbfow

    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    diffusion   = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet  = networks.init_controlnet()

    if args.cnet_ckpt is not None:
        print('Resuming training...')
        controlnet.load_state_dict(torch.load(args.cnet_ckpt))
    else:
        print('Copying weights from diffusion model')
        controlnet.load_state_dict(diffusion.state_dict(), strict=False)

    # freeze the unet weights
    for p in diffusion.parameters():
        p.requires_grad = False

    # Move everything to DEVICE
    autoencoder.to(DEVICE)
    diffusion.to(DEVICE)
    controlnet.to(DEVICE)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=args.lr)

    with torch.no_grad():
        with autocast(enabled=True):
            z = trainset[0]['followup_latent_path']

    scale_factor = 1 / torch.std(z)
    print(f"Scaling factor set to {scale_factor}")

    scheduler = DDPMScheduler(num_train_timesteps=1000, 
                              schedule='scaled_linear_beta', 
                              beta_start=0.0015, 
                              beta_end=0.0205)
    
    writer = SummaryWriter()

    global_counter  = { 'train': 0, 'valid': 0 }
    loaders         = { 'train': train_loader, 'valid': valid_loader }
    datasets        = { 'train': trainset, 'valid': validset }


# Modified training loop to support a tensor of past scans in dim 1
    for epoch in range(args.n_epochs):
        for mode, loader in loaders.items():
            print('mode:', mode)
            controlnet.train() if mode == 'train' else controlnet.eval()
            epoch_loss = 0.0
            progress_bar = tqdm(enumerate(loader), total=len(loader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                if mode == 'train':
                    optimizer.zero_grad(set_to_none=True)

                # Move context to device
                # Expected shape: (batch, 1, covariate_dim)
                context = batch['context'].to(DEVICE).float()

                # Retrieve multi-scan starting latents and ages
                zs_all = batch['starting1_latent_path'].to(DEVICE) * scale_factor
                a1 = batch['starting1_age'].to(DEVICE)

                # Branch by dimension: single-scan vs multi-scan
                if zs_all.dim() == 5:
                    starting_z_all = [zs_all]
                    starting_a_all = [a1]
                elif zs_all.dim() == 6:
                    num_past = zs_all.size(1)
                    as_all = a1.unsqueeze(1).expand(-1, num_past) if a1.dim() == 1 else a1
                    starting_z_all = list(torch.unbind(zs_all, dim=1))
                    starting_a_all = list(torch.unbind(as_all, dim=1))
                else:
                    raise ValueError(f"Unexpected zs_all.dim(): {zs_all.dim()}")

                # Build controlnet conditions
                controlnet_condition_all = []
                n = starting_z_all[0].size(0)
                for z_i, a_i in zip(starting_z_all, starting_a_all):
                    # Ensure z_i is 5D: (batch, C, H, W, D)
                    if z_i.dim() == 4:
                        z_i = z_i.unsqueeze(1)
                    age_map = a_i.view(n, 1, *([1] * (z_i.dim() - 2)))
                    age_map = age_map.expand(n, 1, *z_i.shape[-3:])
                    cond_i = torch.cat([z_i, age_map], dim=1)  # (batch, C+1, H, W, D)
                    controlnet_condition_all.append(cond_i)

                # Prepare follow-up latent and noise
                followup_z = batch['followup_latent_path'].to(DEVICE) * scale_factor
                noise      = torch.randn_like(followup_z)
                timesteps  = torch.randint(
                    0,
                    scheduler.num_train_timesteps,
                    (n,),
                    device=DEVICE
                ).long()
                images_noised = scheduler.add_noise(followup_z, noise=noise, timesteps=timesteps)

                with torch.set_grad_enabled(mode == 'train'):
                    with autocast(enabled=True):
                        prev_noise_pred = None
                        for j, cond in enumerate(controlnet_condition_all):
                            down_h, mid_h = controlnet(
                                x=images_noised.float(),
                                timesteps=timesteps,
                                context=context,
                                controlnet_cond=cond.float()
                            )
                            noise_pred_j = diffusion(
                                x=images_noised.float(),
                                timesteps=timesteps,
                                context=context,
                                down_block_additional_residuals=down_h,
                                mid_block_additional_residual=mid_h
                            )

                            if j == 0:
                                prev_noise_pred = noise_pred_j
                            else:
                                updated = noise_pred_j.clone()
                                norm_updated = updated.view(-1).norm()

                                delta = torch.abs(noise_pred_j - prev_noise_pred)
                                delta = delta / (delta.view(-1).norm() + 1e-8)

                                age_diff = (
                                    starting_a_all[j] - starting_a_all[j-1]
                                ).float().view(n, *([1] * (updated.dim() - 1)))

                                updated += ((delta * noise_pred_j) / (age_diff + 1e-8)) * (j / (j + 1))
                                updated = (updated / (updated.view(-1).norm() + 1e-8)) * norm_updated
                                prev_noise_pred = updated

                        # Compute MSE loss
                        noise_pred = prev_noise_pred
                        loss = F.mse_loss(noise_pred.float(), noise.float())

                if mode == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                # Logging
                writer.add_scalar(f'{mode}/batch-mse', loss.item(), global_counter[mode])
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                global_counter[mode] += 1

            # Epoch end
            epoch_loss /= len(loader)
            writer.add_scalar(f'{mode}/epoch-mse', epoch_loss, epoch)

        # Save checkpoint
        if epoch > 2:
            savepath = os.path.join(args.output_dir, f'cnet-ep-{epoch}.pth')
            torch.save(controlnet.state_dict(), savepath)