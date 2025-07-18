import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from inspect import isfunction
from Diffusion.unet import Score_UNet
from Layers import SpatialTransformer, motion_block
import math

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,  n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas

class Diffusion(nn.Module):
    def __init__(self, inshape, indim):
        super().__init__()

        self.inshape = inshape
        self.unet = Score_UNet(in_channel=2*indim+2, out_channel=2*indim, image_size=inshape[0])
        self.deform = motion_block(input_nc=16)
        self.num_timesteps = 1000
        self.loss_type = 'l2'
        
        self.mu = nn.Conv2d(2,2,kernel_size=3, padding=1)
        self.logvar = nn.Conv2d(2,2,kernel_size=3, padding=1)
        self.F = nn.Conv2d(4,2,kernel_size=3, padding=1)
        # loss
        self.lambda_L = 1
        self.loss_func = nn.L1Loss(reduction='mean') if self.loss_type == 'l1' else nn.MSELoss(reduction='mean')
        self.transformer = SpatialTransformer(inshape, mode='bilinear')

        opt = {"schedule": "linear",
                "n_timestep": 2000,
                "linear_start": 1e-6,
                "linear_end": 1e-2}
        self.set_new_noise_schedule(opt)

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))   # Gaussian noise
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def forward(self, moving, fixed, condition):

        x_start = torch.cat([moving, fixed], dim=1)
        b, c, h, w = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,), device=x_start.device).long()
        noise = default(None, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        Condition = torch.cat([moving, fixed], dim=1) if condition is None else condition
        Condition = F.interpolate(Condition, size=(h,w), mode='bilinear', align_corners=True)
        input = torch.cat([Condition, x_noisy], dim=1)
        x_recon = self.unet(input, t) 
        D = self.deform(x_recon)
        D = self.F(D) 

        l_pix = self.loss_func(x_start, x_recon)
        
        return D, l_pix
    
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):

        score = torch.cat([condition_x, x], dim=1) if condition_x is not None else x
        x_recon = self.unet(score, t)
        # x_recon = self.predict_start_from_noise(x, t=t, noise=score)  # noise

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,*((1,) * (len(x.shape) - 1)))
        
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, moving, fixed, condition=None):

        if len(moving.shape) == 5:
            s,t,c,h,w = moving.shape
            moving = moving.reshape(s*t,c,h,w)
            fixed = fixed.reshape(s*t,c,h,w)
            condition = condition.reshape(s*t,2,self.inshape[0],self.inshape[1])

        elif len(moving.shape) == 4:
            t,c,h,w = moving.shape

        condition = F.interpolate(condition, size=(h,w), mode='bilinear', align_corners=True)

        x_start = torch.cat([moving, fixed], dim=1)
        b, c, h, w = x_start.shape

        fw_timesteps = 7
        bw_timesteps = 70
        t = torch.full((b,), fw_timesteps, dtype=torch.long).cuda()
        with torch.no_grad():
            # ################ Forward ##############################
            d2n_img = self.q_sample(x_start, t)

            # ################ Reverse ##############################
            img = d2n_img   # latent
            ret_img = [d2n_img]

            # for ispr in range(1):
            for i in (reversed(range(0, bw_timesteps))):
                t = torch.full((b,), i, dtype=torch.long).cuda()
                img = self.p_sample(img, t, condition_x=condition)
                # if i % 11 == 0: #
                ret_img.append(img)
        fin_latent = ret_img[-1]

        D = self.deform('udec',fin_latent) # deformation field: net
        D = self.F(D)

        return D

    def predict_noise_from_x_theta(self, x_theta, x_t, t):
        up = x_t - extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_theta
        down = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return up / down
        