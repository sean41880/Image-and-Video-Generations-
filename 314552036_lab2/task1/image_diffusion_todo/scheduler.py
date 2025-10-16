from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

def extract(input, t: torch.Tensor, x: torch.Tensor):
    if t.ndim == 0:
        t = t.unsqueeze(0)
    shape = x.shape
    t = t.long().to(input.device)
    out = torch.gather(input, 0, t)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

class BaseScheduler(nn.Module):
    def __init__(
        self, num_train_timesteps: int, beta_1: float, beta_T: float, mode="linear"
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_T, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (
                torch.linspace(beta_1**0.5, beta_T**0.5, num_train_timesteps) ** 2
            )
        elif mode == "cosine":
            ######## TODO ########
            # Implement the cosine beta schedule (Nichol & Dhariwal, 2021).
            # Hint:
            # 1. Define alphā_t = f(t/T) where f is a cosine schedule:
            #       alphā_t = cos^2( ( (t/T + s) / (1+s) ) * (π/2) )
            #    with s = 0.008 (a small constant for stability).
            # 2. Convert alphā_t into betas using:
            #       beta_t = 1 - alphā_t / alphā_{t-1}
            # 3. Return betas as a tensor of shape [num_train_timesteps].
            s = 0.008
            steps = num_train_timesteps
            t_arr = torch.arange(steps + 1, dtype=torch.float64)
            f = lambda t: torch.cos(((t / steps + s) / (1 + s)) * np.pi / 2) ** 2
            alphas_cumprod = f(t_arr)
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
            betas = betas.to(torch.float32)
               
        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def uniform_sample_t(
        self, batch_size, device: Optional[torch.device] = None
    ) -> torch.IntTensor:
        """
        Uniformly sample timesteps.
        """
        ts = np.random.choice(np.arange(self.num_train_timesteps), batch_size)
        ts = torch.from_numpy(ts)
        if device is not None:
            ts = ts.to(device)
        return ts

class DDPMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode="linear",
        sigma_type="small",
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        
        self.schedule_mode = mode      

        # sigmas correspond to $\sigma_t$ in the DDPM paper.
        self.sigma_type = sigma_type
        if sigma_type == "small":
            # when $\sigma_t^2 = \tilde{\beta}_t$.
            alphas_cumprod_t_prev = torch.cat(
                [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
            )
            sigmas = (
                (1 - alphas_cumprod_t_prev) / (1 - self.alphas_cumprod) * self.betas
            ) ** 0.5
        elif sigma_type == "large":
            # when $\sigma_t^2 = \beta_t$.
            sigmas = self.betas ** 0.5

        self.register_buffer("sigmas", sigmas)

    
    
    def step(self, x_t: torch.Tensor, t: int, net_out: torch.Tensor, predictor: str):
        if predictor == "noise": #### TODO
            return self.step_predict_noise(x_t, t, net_out)
        elif predictor == "x0": #### TODO
            return self.step_predict_x0(x_t, t, net_out)
        elif predictor == "mean": #### TODO
            return self.step_predict_mean(x_t, t, net_out)
        else:
            raise ValueError(f"Unknown predictor: {predictor}")

    
    def step_predict_noise(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor):
        """
        Noise prediction version (the standard DDPM formulation).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            eps_theta: predicted noise ε̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        # 1. Extract beta_t, alpha_t, and alpha_bar_t from the scheduler.
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        alpha_bar_t_prev = extract(torch.cat([self.alphas_cumprod.new_ones(1), self.alphas_cumprod[:-1]], 0), t, x_t)
        # 2. Compute the predicted mean μ_θ(x_t, t) = 1/√α_t * (x_t - (β_t/√(1-ᾱ_t)) * ε̂_θ).
        mean = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps_theta)
        # 3. Compute the posterior variance \tilde{β}_t = ((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t.
        posterior_var = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
        # 4. Add Gaussian noise scaled by √(\tilde{β}_t) unless t == 0.
        noise = torch.randn_like(x_t, device=x_t.device) if (t > 0).any() else torch.zeros_like(x_t, device=x_t.device)
        mask = (t != 0).float().reshape(-1, *([1] * (x_t.dim() - 1))).to(x_t.device)
        sample_prev = mean + mask * torch.sqrt(posterior_var) * noise
        # 5. Return the final sample at t-1.
        #######################
        return sample_prev

    
    def step_predict_x0(self, x_t: torch.Tensor, t: int, x0_pred: torch.Tensor):
        """
        x0 prediction version (alternative DDPM objective).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            x0_pred: predicted clean image x̂₀(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        alpha_bar_t_prev = extract(torch.cat([self.alphas_cumprod.new_ones(1), self.alphas_cumprod[:-1]], 0), t, x_t)
        # Posterior mean (from DDPM paper)
        mean = (
            torch.sqrt(alpha_bar_t_prev) * beta_t * x0_pred + torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) * x_t
        ) / (1 - alpha_bar_t)
        posterior_var = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
        noise = torch.randn_like(x_t, device=x_t.device) if (t > 0).any() else torch.zeros_like(x_t, device=x_t.device)
        mask = (t != 0).float().reshape(-1, *([1] * (x_t.dim() - 1))).to(x_t.device)
        sample_prev = mean + mask * torch.sqrt(posterior_var) * noise
        #######################
        return sample_prev

    
    def step_predict_mean(self, x_t: torch.Tensor, t: int, mean_theta: torch.Tensor):
        """
        Mean prediction version (directly outputting the posterior mean).
        
        Input:
            x_t: noisy image at timestep t
            t: current timestep
            mean_theta: network-predicted posterior mean μ̂_θ(x_t, t)
        Output:
            sample_prev: denoised image sample at timestep t-1
        """
        ######## TODO ########
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        beta_t = extract(self.betas, t, x_t)
        alpha_t = extract(self.alphas, t, x_t)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        alpha_bar_t_prev = extract(torch.cat([self.alphas_cumprod.new_ones(1), self.alphas_cumprod[:-1]], 0), t, x_t)
        posterior_var = ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)) * beta_t
        noise = torch.randn_like(x_t, device=x_t.device) if (t > 0).any() else torch.zeros_like(x_t, device=x_t.device)
        mask = (t != 0).float().reshape(-1, *([1] * (x_t.dim() - 1))).to(x_t.device)
        sample_prev = mean_theta + mask * torch.sqrt(posterior_var) * noise
        #######################
        return sample_prev

    
    
    # https://nn.labml.ai/diffusion/ddpm/utils.html
    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor): # get t th const 
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        t: torch.IntTensor,
        eps: Optional[torch.Tensor] = None,
    ):
        """
        A forward pass of a Markov chain, i.e., q(x_t | x_0).

        Input:
            x_0 (`torch.Tensor [B,C,H,W]`): samples from a real data distribution q(x_0).
            t: (`torch.IntTensor [B]`)
            eps: (`torch.Tensor [B,C,H,W]`, optional): if None, randomly sample Gaussian noise in the function.
        Output:
            x_t: (`torch.Tensor [B,C,H,W]`): noisy samples at timestep t.
            eps: (`torch.Tensor [B,C,H,W]`): injected noise.
        """
    
        if eps is None:
            eps       = torch.randn(x_0.shape, device=x_0.device)

        ######## TODO ########
        # DO NOT change the code outside this part.
        # Assignment 1. Implement the DDPM forward step.
        alpha_bar_t = extract(self.alphas_cumprod, t, x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
        #######################

        return x_t, eps

class DDIMScheduler(BaseScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float,
        beta_T: float,
        mode: str = "linear",
        num_inference_timesteps: int = 50,
        eta: float = 0.0,
    ):
        super().__init__(num_train_timesteps, beta_1, beta_T, mode)
        self.eta = float(eta)
        self.set_inference_timesteps(num_inference_timesteps)

    def set_inference_timesteps(self, num_inference_timesteps: int):
        """
        Define the inference schedule (a subset of training timesteps, descending order).
        Inputs:
            num_inference_timesteps (int): number of inference steps (e.g., 50).
        """
        ######## TODO ########
        # Hint:
        #   - Define the DDIM inference schedule based on the given num_inference_timesteps.
        #   - The schedule should be a subset of training timesteps, ordered in descending fashion.
        #   - Store the result in `self.timesteps` (as a torch tensor) 
        #   - Store the step ratio in `self._ddim_step_ratio` for later use when computing previous t.
        #   - Compute a `step_ratio` that maps inference steps to training steps.
        # DO NOT change the code outside this part.
        step_ratio = self.num_train_timesteps // num_inference_timesteps
        timesteps = (np.arange(0, num_inference_timesteps) * step_ratio).round()[::-1].copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps)
        self._ddim_step_ratio = step_ratio
        #######################

    def _get_teeth(self, consts: torch.Tensor, t: torch.Tensor):
        const = consts.gather(-1, t)
        return const.reshape(-1, 1, 1, 1)

    @torch.no_grad()
    def step(self, x_t: torch.Tensor, t: int, eps_theta: torch.Tensor, predictor: str):
        """
        One step DDIM update: x_t -> x_{t_prev} with deterministic/stochastic control via eta.

        Input:
            x_t: [B,C,H,W]
            t: current absolute timestep index
            eps_theta: predicted noise
            predictor: predictor type
        Output:
            sample_prev: x at previous inference timestep
        """
        ######## TODO ########
        # DO NOT change the code outside this part.
        assert predictor == "noise", "In assignment 2, we only implement DDIM with noise predictor."
        # DDIM公式
        if isinstance(t, int):
            t = torch.tensor([t], device=x_t.device)
        step_ratio = self._ddim_step_ratio
        t_prev = t - step_ratio
        t_prev = torch.clamp(t_prev, min=0)
        alpha_bar_t = extract(self.alphas_cumprod, t, x_t)
        alpha_bar_t_prev = extract(self.alphas_cumprod, t_prev, x_t)
        # 還原 x0
        x0_pred = (x_t - (1 - alpha_bar_t).sqrt() * eps_theta) / alpha_bar_t.sqrt()
        # sigma (stochasticity)
        if self.eta > 0:
            sigma = self.eta * ((1 - alpha_bar_t_prev) / (1 - alpha_bar_t)).sqrt() * (1 - alpha_bar_t / alpha_bar_t_prev).sqrt()
            noise = torch.randn_like(x_t)
        else:
            sigma = 0
            noise = 0
        sample_prev = alpha_bar_t_prev.sqrt() * x0_pred + (1 - alpha_bar_t_prev).sqrt() * eps_theta + sigma * noise
        #######################
        return sample_prev
