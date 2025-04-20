import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
from generative.networks.schedulers import DDIMScheduler
from tqdm import tqdm

from . import utils
from . import const


@torch.no_grad()
def sample_using_diffusion(
    autoencoder: nn.Module, 
    diffusion: nn.Module, 
    context: torch.Tensor,
    device: str, 
    scale_factor: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True
) -> torch.Tensor: 
    """
    Sampling random brain MRIs that follow the covariates in `context`.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.
    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    context = context.unsqueeze(0).to(device).to(device)

    # drawing a random z_T ~ N(0,I)
    z = torch.randn(const.LATENT_SHAPE_DM).unsqueeze(0).to(device)
    
    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps
    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                timestep = torch.tensor([t]).to(device)
                
                # predict the noise
                noise_pred = diffusion(
                    x=z.float(), 
                    timesteps=timestep, 
                    context=context.float(), 
                )

                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)
    
    # decode the latent
    z = z / scale_factor
    z = utils.to_vae_latent_trick( z.squeeze(0).cpu() )
    x = autoencoder.decode_stage_2_outputs( z.unsqueeze(0).to(device) )
    x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x


@torch.no_grad()
def sample_using_controlnet_and_z(
    autoencoder: nn.Module, 
    diffusion: nn.Module,
    controlnet: nn.Module,
    starting_z_all: torch.Tensor,
    starting_a_all: int, 
    context: torch.Tensor, 
    device: str,
    scale_factor: int = 1,
    average_over_n: int = 1,
    num_training_steps: int = 1000,
    num_inference_steps: int = 50,
    schedule: str = 'scaled_linear_beta',
    beta_start: float = 0.0015, 
    beta_end: float = 0.0205, 
    verbose: bool = True
) -> torch.Tensor:
    """
    The inference process described in the paper.

    Args:
        autoencoder (nn.Module): the KL autoencoder
        diffusion (nn.Module): the UNet 
        controlnet (nn.Module): the ControlNet
        starting_z_all (list of torch.Tensor): the latent from the MRI of the starting visit (for all past scans)
        starting_a_all (list of int): the starting ages (for all past scans)
        context (torch.Tensor): the covariates
        device (str): the device ('cuda' or 'cpu')
        scale_factor (int, optional): the scale factor (see Rombach et Al, 2021). Defaults to 1.
        average_over_n (int, optional): LAS parameter m. Defaults to 1.
        num_training_steps (int, optional): T parameter. Defaults to 1000.
        num_inference_steps (int, optional): reduced T for DDIM sampling. Defaults to 50.
        schedule (str, optional): noise schedule. Defaults to 'scaled_linear_beta'.
        beta_start (float, optional): noise starting level. Defaults to 0.0015.
        beta_end (float, optional): noise ending level. Defaults to 0.0205.
        verbose (bool, optional): print progression bar. Defaults to True.

    Returns:
        torch.Tensor: the inferred follow-up MRI
    """
    # Using DDIM sampling from (Song et al., 2020) allowing for a 
    # deterministic reverse diffusion process (except for the starting noise)
    # and a faster sampling with fewer denoising steps.
    scheduler = DDIMScheduler(num_train_timesteps=num_training_steps,
                              schedule=schedule,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              clip_sample=False)

    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    controlnet_condition_all = []
    
    for i in range(len(starting_z_all)):
        # preparing controlnet spatial condition.
        starting_z             = starting_z_all[i].unsqueeze(0).to(device)
        concatenating_age      = torch.tensor([ starting_a_all[i] ]).view(1, 1, 1, 1, 1).expand(1, 1, *starting_z.shape[-3:]).to(device)
        controlnet_condition   = torch.cat([ starting_z, concatenating_age ], dim=1).to(device)
        controlnet_condition_all.append(controlnet_condition)
        print(f"strating z at {i} :{starting_z_all[i].shape}")
        print(f"strating controlnet_condition at {i} :{controlnet_condition_all[i].shape}")
        

    # the subject-specific variables and the progression-related 
    # covariates are concatenated into a vector outside this function. 
    context = context.unsqueeze(0).unsqueeze(0).to(device)

    if average_over_n > 1:
        context = context.repeat(average_over_n, 1, 1)

    for i in range(len(controlnet_condition_all)):
        # if performing LAS, we repeat the inputs for the diffusion process
        # m times (as specified in the paper) and perform the reverse diffusion
        # process in parallel to avoid overheads.
        controlnet_condition_all[i]  = controlnet_condition_all[i].repeat(average_over_n, 1, 1, 1, 1) 
        print(f"strating controlnet_condition at {i} :{controlnet_condition_all[i].shape}")

    print(f"controlnet condition length: {len(controlnet_condition_all)}")
    # print(f"controlnet condition shape of each: {controlnet_condition_all[0].shape}")
    
    # this is z_T - the starting noise.
    z = torch.randn(average_over_n, *starting_z_all[0].shape).to(device)
    print(f"Z shape: {z.shape}")
    z_original_norm = z.view(-1).norm()
    # z = torch.randn(average_over_n, *starting_z_all[-1].shape[1:]).to(device) 
    z += (starting_z_all[-1].unsqueeze(0))
    z = z / z.view(-1).norm()
    z = z * z_original_norm


    progress_bar = tqdm(scheduler.timesteps) if verbose else scheduler.timesteps

    for t in progress_bar:
        with torch.no_grad():
            with autocast(enabled=True):

                # convert the timestep to a tensor.
                timestep = torch.tensor([t]).repeat(average_over_n).to(device)

                # get the intermediate features from the ControlNet
                # by feeding the starting latent, the covariates and the timestep
                prev_noise_step = -1
                # z_same = z.clone()

                for j in range(len(controlnet_condition_all)):
                    down_h, mid_h = controlnet(
                        x=z.float(), 
                        timesteps=timestep, 
                        context=context,
                        controlnet_cond=controlnet_condition_all[j].float()
                    )
                    # print(f"efdaHJEwlfhkv;cekJFK;eribgjlakejlfbgjaerbfljkaeb;gkaer: {z.shape}")

                    # the diffusion takes the intermediate features and predicts
                    # the noise. This is why we conceptualize the two networks as
                    # as a unified network.
                    noise_pred = diffusion(
                        x=z.float(), 
                        timesteps=timestep, 
                        context=context.float(), 
                        down_block_additional_residuals=down_h,
                        mid_block_additional_residual=mid_h
                    )

                    if(j == 0):
                        prev_noise_step = noise_pred
                    else:
                        updated_curr_tensor = noise_pred.clone()
                        updated_curr_tensor_norm = updated_curr_tensor.view(-1).norm()
                        temp_latent = torch.abs(noise_pred - prev_noise_step)
                        temp_latent = temp_latent / (temp_latent.view(-1).norm() + 1e-8)  # norm over all dims

                        updated_curr_tensor += (((temp_latent * noise_pred) / ((starting_a_all[j] - starting_a_all[j-1])*1.0))*(j/(j+1)))
                        updated_curr_tensor = (updated_curr_tensor  / (updated_curr_tensor.view(-1).norm() + 1e-8)) * updated_curr_tensor_norm
                        prev_noise_step = updated_curr_tensor

                noise_pred = prev_noise_step
                # the scheduler applies the formula to get the 
                # denoised step z_{t-1} from z_t and the predicted noise
                z, _ = scheduler.step(noise_pred, t, z)

    # Here we conclude Latent Average Stabilization by averaging 
    # m different latents from m different samplings.
    z = (z / scale_factor).sum(axis=0) / average_over_n
    z = utils.to_vae_latent_trick(z.squeeze(0).cpu())

    # decode the latent using the Decoder block from the KL autoencoder.
    x = autoencoder.decode_stage_2_outputs( z.unsqueeze(0).to(device) )
    x = utils.to_mni_space_1p5mm_trick( x.squeeze(0).cpu() ).squeeze(0)
    return x