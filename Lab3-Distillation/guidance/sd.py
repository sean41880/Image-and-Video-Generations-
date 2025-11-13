from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from peft import LoraConfig


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] Loading Stable Diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet.eval()
        #-------------------------------
        self.unet.enable_gradient_checkpointing()
        self.unet.to(self.dtype)
        #-------------------------------

        # Freeze models
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Schedulers
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.inverse_scheduler.alphas_cumprod = self.inverse_scheduler.alphas_cumprod.to(self.device)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod
        
        print(f'[INFO] Loaded Stable Diffusion!')
        
        # Initialize VSD components if needed
        if args.loss_type == "vsd":
            self._init_vsd_components(args.lora_rank)
    
    def _init_vsd_components(self, lora_rank=4):
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")

        # --- Teacher UNet: No LoRA, Frozen --------------------------------------
        import copy
        self.unet_teacher = copy.deepcopy(self.unet).eval().requires_grad_(False)
        self.unet_teacher = self.unet_teacher.to(self.dtype)

        # --- Student UNet: Add LoRA and Trainable --------------------------------
        self.unet.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

        # Memory Optimization Option:
        self.unet.enable_gradient_checkpointing()



    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """Get text embeddings from prompt"""
        inputs = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=7.5):
        """
        Predict noise with Classifier-Free Guidance (CFG)
        
        CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
        """
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred
    
    def get_sds_loss(self, latents, text_embeddings, guidance_scale=7.5):
        """
        Score Distillation Sampling (SDS) Loss
        
        Reference: DreamFusion (https://arxiv.org/abs/2209.14988)
        """
        # TODO: Implement SDS loss
        # --------------------------------------------------------------------
        # latents: (B, C, H, W) assumed in same scale used by UNet (usually in latent space)
        device = latents.device
        dtype = latents.dtype
        B = latents.shape[0]

        # 1. Random timestep t per batch element
        t_int = torch.randint(self.min_step, self.max_step + 1, (B,), device=device)
        alphas_t = self.alphas[t_int].to(device=device, dtype=dtype)
        sqrt_alpha = torch.sqrt(alphas_t).view(B, 1, 1, 1)
        sqrt_one_minus_alpha = torch.sqrt(1.0 - alphas_t).view(B, 1, 1, 1)

        # 2. Add Gaussian noise to latents
        noise = torch.randn_like(latents)
        latents_noisy = sqrt_alpha * latents + sqrt_one_minus_alpha * noise

        # 3. Diffusion model predicts noise under classifier-free guidance
        noise_pred = self.get_noise_preds(latents_noisy, t_int, text_embeddings, guidance_scale=guidance_scale)

        # 4. Compute the SDS gradient target
        grad = (noise_pred - noise).detach()  # no grad into diffusion model

        # 5. Pseudo-loss whose gradient = SDS gradient
        #    d(loss)/d(latents) = grad
        loss = (latents * grad).sum() / B
        # --------------------------------------------------------------------

        return loss

    
    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) Loss
        
        Reference: ProlificDreamer (https://arxiv.org/abs/2305.16213)
        """
        # TODO: Implement VSD loss
        # --------------------------------------------------------------------
        device = self.device
        B = latents.shape[0]
        assert text_embeddings.shape[0] == 2 * B, "[VSD] text_embeddings 需為 [2B, ...]。"

        # sample t
        t = torch.randint(self.min_step, self.max_step + 1, (B,), device=device).long()

        # noisy
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        # student prediction (LoRA-enabled)
        noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)

        # scheduler prediction type
        pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        a_bar = self.alphas[t].view(-1, 1, 1, 1)
        sqrt_ab = a_bar.sqrt()
        sqrt_omab = (1.0 - a_bar).sqrt()

        # convert v → ε if needed
        if pred_type == "epsilon":
            eps_student = noise_pred
        elif pred_type == "v_prediction":
            eps_student = sqrt_ab * noise_pred + sqrt_omab * latents_noisy
        else:
            eps_student = noise_pred

        # ----------------- **VSD 核心: 老師模型 (不學習) 預測** -----------------
        with torch.no_grad():
            # Teacher forward must match student CFG input format
            latents_noisy_teacher = latents_noisy.half()
            t_teacher = t
            text_emb_teacher = text_embeddings.half()

            # 1) duplicate input, same as student CFG
            latent_in = torch.cat([latents_noisy_teacher] * 2)
            tt = torch.cat([t_teacher] * 2)

            # 2) teacher UNet forward
            noise_pred_teacher = self.unet_teacher(latent_in, tt, encoder_hidden_states=text_emb_teacher).sample

            # 3) split (uncond, cond)
            noise_pred_teacher_uncond, noise_pred_teacher_pos = noise_pred_teacher.chunk(2)

            # 4) same CFG formula
            noise_teacher = noise_pred_teacher_uncond + guidance_scale * (noise_pred_teacher_pos - noise_pred_teacher_uncond)

            # 5) convert v → ε if needed
            if pred_type == "epsilon":
                eps_teacher = noise_teacher
            elif pred_type == "v_prediction":
                eps_teacher = sqrt_ab * noise_teacher + sqrt_omab * latents_noisy
            else:
                eps_teacher = noise_teacher


        # ----------------- **VSD 主 Loss: 比較 student vs. teacher** -----------------
        w = (1.0 - a_bar)
        main_loss = ((w * (eps_student - eps_teacher)) ** 2).mean()

        # ----------------- **LoRA L2 regularization (保留原寫法)** -----------------
        l2 = 0.0
        if hasattr(self, "lora_layers") and self.lora_layers:
            for p in self.lora_layers:
                if p.requires_grad and p.dtype == latents.dtype:
                    l2 = l2 + (p ** 2).sum()
            l2 = l2 * float(lora_loss_weight)

        loss = main_loss + l2
        return loss
        # --------------------------------------------------------------------


    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM Inversion: x0 -> x_t
        
        Inverts clean latents (x0) to noisy latents (x_t) using DDIM inversion.
        
        Args:
            latents: Clean latents x0
            target_t: Target timestep to invert to
            text_embeddings: Text condition
            guidance_scale: CFG scale (typically negative for inversion!)
            n_steps: Number of inversion steps
            eta: Noise level for stochasticity
            
        Returns:
            Inverted noisy latents x_t
        """
        # TODO: (Implement DDIM inversion by yourself — do NOT call built-in inversion helpers):
        # --------------------------------------------------------------------
        device = self.device
        x = latents.clone()

        # Support int or tensor for target_t
        if isinstance(target_t, int):
            t_target = torch.full((latents.shape[0],), target_t, device=device, dtype=torch.long)
        else:
            t_target = target_t.to(device).long()

        # Build a simple linear grid from 0 -> max(t_target) with n_steps
        t_vals = torch.linspace(0, float(t_target.max().item()), steps=n_steps + 1, device=device)
        t_vals = t_vals[1:]  # skip 0 since x is x0 already

        for ti in t_vals:
            ti_int = torch.clamp(ti.round().long(), min=0, max=self.num_train_timesteps - 1)
            t = torch.full((latents.shape[0],), ti_int.item(), device=device, dtype=torch.long)

            a_bar = self.alphas[t].view(-1, 1, 1, 1)                  # alpha_bar_t
            a_bar_prev = self.alphas[torch.clamp(t - 1, 0)].view(-1, 1, 1, 1)

            # Predict eps with (usually negative) guidance scale for inversion
            eps_pred = self.get_noise_preds(x, t, text_embeddings, guidance_scale)

            pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
            if pred_type == "epsilon":
                eps_t = eps_pred
            elif pred_type == "v_prediction":
                eps_t = a_bar.sqrt() * eps_pred + (1.0 - a_bar).sqrt() * x
            else:
                eps_t = eps_pred

            # Estimate x0 from current x_t and eps_t (DDIM relation)
            x0_pred = (x - (1.0 - a_bar).sqrt() * eps_t) / (a_bar.sqrt().clamp_min(1e-8))

            # Compute DDIM forward step to a noisier time
            a_bar_next = a_bar
            sigma_t = eta * torch.sqrt(
                ((1.0 - a_bar_prev) / (1.0 - a_bar_next).clamp_min(1e-8)) * (1.0 - (a_bar_next / a_bar_prev).clamp_min(1e-8))
            ).clamp_min(0.0)

            coeff = torch.sqrt((1.0 - a_bar_next - sigma_t ** 2).clamp_min(0.0))
            z = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            x = a_bar_next.sqrt() * x0_pred + coeff * eps_t + sigma_t * z

        return x
        # --------------------------------------------------------------------

    
    def get_sdi_loss(
        self, 
        latents,                    
        text_embeddings,            
        guidance_scale=7.5,         
        current_iter=0,             
        total_iters=500,            
        inversion_guidance_scale=-7.5,  
        inversion_n_steps=10,       
        inversion_eta=0.3,          
        update_interval=25,        
    ):
        """
        Score Distillation via Inversion (SDI) Loss
        
        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)
        
        Key Insight: Instead of using random noise like SDS, SDI uses DDIM inversion
        to get better noise that's consistent with the current latents.
        
        Strategy:
        1. Timestep annealing: t decreases from max_step to min_step during training
        2. Periodically update target via DDIM inversion
        3. Use MSE loss between current latents and cached target
        
        Args:
            latents: Current optimized latents (B, 4, H, W)
            text_embeddings: Concatenated [uncond, cond] embeddings (2*B, seq_len, dim)
            guidance_scale: CFG scale for final denoising step
            current_iter: Current training iteration number
            total_iters: Total number of training iterations
            inversion_guidance_scale: CFG scale for DDIM inversion (typically negative)
            inversion_n_steps: Number of inversion steps from x0 to x_t
            inversion_eta: Stochasticity level for DDIM inversion (0=deterministic)
            update_interval: Update cached target every N iterations
            
        Returns:
            loss: MSE loss between current latents and cached target
        """
        B = latents.shape[0]
        
        # TODO: Create current timestep tensor based on training progress
        # t = ...
        # --------------------------------------------------------------------
        prog = float(current_iter) / max(1, int(total_iters))
        t_scalar = int(round(self.max_step - (self.max_step - self.min_step) * prog))
        t = torch.full((B,), t_scalar, device=self.device, dtype=torch.long)
        # --------------------------------------------------------------------
        # Check if we need to update target
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')
        
        if should_update:
            with torch.no_grad():
                # Perform DDIM inversion: x0 -> x_t
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )
                
                # TODO: Predict noise from inverted noisy latents
                # noise_pred = ...
                # ----------------------------------------------------------------
                noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)
                # ----------------------------------------------------------------
                
                # TODO: Denoise to get target x0 using predicted noise
                # target = ...
                # ----------------------------------------------------------------
                a_bar = self.alphas[t].view(-1, 1, 1, 1)
                pred_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
                if pred_type == "epsilon":
                    eps_pred = noise_pred
                elif pred_type == "v_prediction":
                    eps_pred = a_bar.sqrt() * noise_pred + (1.0 - a_bar).sqrt() * latents_noisy
                else:
                    eps_pred = noise_pred

                target = (latents_noisy - (1.0 - a_bar).sqrt() * eps_pred) / (a_bar.sqrt().clamp_min(1e-8))
                # ----------------------------------------------------------------
                
                # Cache the target
                self.sdi_target = target.detach()
        
        # TODO: Compute MSE loss between current latents and cached target
        # loss = ...
        # --------------------------------------------------------------------
        loss = F.mse_loss(latents, self.sdi_target)
        # --------------------------------------------------------------------

        return loss
        
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode RGB images to latents"""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents