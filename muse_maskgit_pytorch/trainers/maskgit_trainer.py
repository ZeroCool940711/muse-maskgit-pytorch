from typing import List
import os
import torch  # noqa: F401
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.optimization import SchedulerType
from ema_pytorch import EMA
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from ema_pytorch import EMA
from PIL import Image
from diffusers.optimization import get_scheduler
from muse_maskgit_pytorch.t5 import t5_encode_text_from_encoded
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import BaseAcceleratedTrainer
from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import (
    BaseAcceleratedTrainer,
    get_optimizer,
)
from tqdm import tqdm
from omegaconf import OmegaConf


try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
except ImportError:
    torch_xla = None
    xm = None
    met = None


class MaskGitTrainer(BaseAcceleratedTrainer):
    def __init__(
        self,
        maskgit: MaskGit,
        dataloader: DataLoader,
        valid_dataloader: DataLoader,
        accelerator: Accelerator,
        optimizer: Optimizer,
        scheduler: SchedulerType,
        *,
        current_step: int,
        num_train_steps: int,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = None,
        save_results_every: int = 100,
        save_model_every: int = 1000,
        log_metrics_every: int = 10,
        results_dir="./results",
        logging_dir="./results/logs",
        apply_grad_penalty_every=4,
        lr=3e-4,
        lr_scheduler_type="constant",
        lr_warmup_steps=500,
        num_cycles=1,
        scheduler_power=1.0,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=0,
        ema_update_every=1,
        validation_prompts=["a photo of a dog"],
        clear_previous_experiments=False,
        validation_image_scale: float = 1.0,
        only_save_last_checkpoint=False,
        use_profiling=False,
        profile_frequency=1,
        row_limit=10,
        optimizer="Lion",
        weight_decay=0.0,
        use_8bit_adam=False,
        args=None,
    ):
        super().__init__(
            dataloader=dataloader,
            valid_dataloader=valid_dataloader,
            accelerator=accelerator,
            current_step=current_step,
            num_train_steps=num_train_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_results_every=save_results_every,
            save_model_every=save_model_every,
            results_dir=results_dir,
            logging_dir=logging_dir,
            apply_grad_penalty_every=apply_grad_penalty_every,
            clear_previous_experiments=clear_previous_experiments,
            validation_image_scale=validation_image_scale,
            only_save_last_checkpoint=only_save_last_checkpoint,
            use_profiling=use_profiling,
            profile_frequency=profile_frequency,
            row_limit=row_limit,
        )
        self.save_results_every = save_results_every
        self.log_metrics_every = log_metrics_every
        self.batch_size = batch_size
        self.current_step = current_step
        self.counter_1 = 0
        self.counter_2 = 0

        # arguments used for the training script,
        # we are going to use them later to save them to a config file.
        self.args = args

        # maskgit
        maskgit.vae.requires_grad_(False)
        maskgit.transformer.t5.requires_grad_(False)
        self.model: MaskGit = maskgit

        self.optim: Optimizer = optimizer
        self.lr_scheduler: SchedulerType = scheduler
        self.model = maskgit
        self.model.vae.requires_grad_(False)
        self.model.transformer.t5.requires_grad_(False)

        all_parameters = set(maskgit.parameters())
        # don't train the vae

        vae_parameters = set(self.model.vae.parameters())
        t5_parameters = set(self.model.transformer.t5.parameters())
        transformer_parameters = all_parameters - vae_parameters - t5_parameters
        self.optim = get_optimizer(use_8bit_adam, optimizer, transformer_parameters, lr, weight_decay)

        self.lr_scheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.num_train_steps * self.gradient_accumulation_steps,
            num_cycles=num_cycles,
            power=scheduler_power,
        )

        # prepare with accelerator

        (
            self.model,
            self.optim,
            self.dl,
            self.valid_dl,
            self.lr_scheduler,
        ) = self.prepare(
            self.model, self.optim, self.dl, self.valid_dl, self.lr_scheduler
        )

        self.use_ema = use_ema
        self.validation_prompts: List[str] = validation_prompts
        if use_ema:
            ema_model = EMA(
                self.model,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
            self.ema_model = ema_model
        else:
            self.ema_model = None

    def save_validation_images(
        self, validation_prompts, step: int, cond_image=None, cond_scale=3, temperature=1
    ):
        images = self.model.generate(
            validation_prompts,
            cond_images=cond_image,
            cond_scale=cond_scale,
            temperature=temperature,
        )
        #step = int(step.item())
        save_file = str(self.results_dir / f"MaskGit" / f"maskgit_{step}.png")
        os.makedirs(str(self.results_dir / f"MaskGit"), exist_ok=True)

        save_image(images, save_file)
        super().log_validation_images(
            [Image.open(save_file)], step, ["|".join(validation_prompts)]
        )

    def train_step(self):
        steps = int(self.steps.item() + (self.batch_size * self.gradient_accumulation_steps))
        apply_grad_penalty = not (steps % self.apply_grad_penalty_every)

        # create two tqdm objects, one for showing the progress bar
        # and another one for showing any extra information we want to show on a different line.
        pbar = tqdm(initial=int(self.steps.item()), total=self.num_train_steps)
        info_bar = tqdm(total=0, bar_format='{desc}')
        #profiling_bar = tqdm(total=0, bar_format='{desc}')


        # use pytorch built-in profiler to gather information on the training for improving performance later.
        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        if self.use_profiling:
            prof = torch.autograd.profiler.profile(use_cuda=True)
            prof.__enter__()
            counter = 1

        if self.use_ema:
            ema_model = self.ema_model.module if self.is_distributed else self.ema_model
        self.model.train()

        if self.accelerator.is_main_process:
            proc_label = f"[P{self.accelerator.process_index:03d}][Master]"
        else:
            proc_label = f"[P{self.accelerator.process_index:03d}][Worker]"

        # logs
        while int(self.steps.item()) < self.num_train_steps:
            # update the tqdm progress bar
            pbar.update(self.batch_size * self.gradient_accumulation_steps)

            for imgs, input_ids, attn_mask in iter(self.dl):
                train_loss = 0.0
                steps = int(self.steps.item() + (self.batch_size * self.gradient_accumulation_steps))

                with torch.no_grad():
                    text_embeds = t5_encode_text_from_encoded(
                        input_ids, attn_mask, self.model.transformer.t5, self.accelerator.device
                    )

                with self.accelerator.accumulate(self.model), self.accelerator.autocast():
                    loss = self.model(imgs, text_embeds=text_embeds)
                    self.accelerator.backward(loss)
                    if self.max_grad_norm is not None and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optim.step()
                    self.lr_scheduler.step()
                    self.optim.zero_grad()

                    if self.use_ema:
                        self.ema_model.update()

                    gathered_loss = self.accelerator.gather_for_metrics(loss)
                    train_loss = gathered_loss.mean() / self.gradient_accumulation_steps

                    logs = {"loss": train_loss, "lr": self.lr_scheduler.get_last_lr()[0]}
                    self.print(f"[S{steps - 1}]{proc_label}: maskgit loss: {logs['loss']} - lr: {logs['lr']}")
                    info_bar.set_description_str(f"[S{steps - 1}]{proc_label}: maskgit loss: {logs['loss']} - lr: {logs['lr']}")
                    self.accelerator.log(logs, step=steps - 1)

                # we made two counters, one for the results and one for the
                # model so we can properly save them without any error
                # no matter what batch_size and gradient_accumulation_steps we use.
                self.counter_1 += (self.batch_size * self.gradient_accumulation_steps)
                self.counter_2 += (self.batch_size * self.gradient_accumulation_steps)

                # save model every so often
                logs['save_model_every'] = ''
                if steps != self.current_step:
                    if self.counter_2 == self.save_model_every or steps == self.num_train_steps:
                        self.accelerator.print(f"[S{steps - 1}]{proc_label}: saving model to {self.results_dir}")

                        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                        maskgit_save_name = "maskgit_superres" if self.model.cond_image_size else "maskgit"
                        file_name = (
                            f"{maskgit_save_name}.{steps - 1}.pt"
                            if not self.only_save_last_checkpoint
                            else f"{maskgit_save_name}.pt"
                        )

                        model_path = self.results_dir.joinpath(file_name)
                        self.accelerator.wait_for_everyone()
                        self.accelerator.save(state_dict, model_path)

                        if self.args and not self.args.do_not_save_config:
                            # save config file next to the model file.
                            conf = OmegaConf.create(vars(self.args))
                            OmegaConf.save(conf, f"{model_path}.yaml")

                        if self.use_ema:
                            self.accelerator.print(
                                f"[S{steps - 1}]{proc_label}: saving EMA model to {self.results_dir}"
                            )
                            ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
                            file_name = (
                                f"{maskgit_save_name}.{steps - 1}.ema.pt"
                                if not self.only_save_last_checkpoint
                                else f"{maskgit_save_name}.ema.pt"
                            )
                            model_path = str(self.results_dir / file_name)
                            self.accelerator.wait_for_everyone()
                            self.accelerator.save(ema_state_dict, model_path)

                            if self.args and not self.args.do_not_save_config:
                                # save config file next to the model file.
                                conf = OmegaConf.create(vars(self.args))
                                OmegaConf.save(conf, f"{model_path}.yaml")

                        self.counter_2 = 0
                # sample results every so often
                logs['save_results_every'] = ''
                if self.counter_1 == self.save_results_every:
                    cond_image = None
                    if self.model.cond_image_size:
                        self.accelerator.print(
                            "With conditional image training, we set the validation prompts to empty strings"
                        )
                        cond_image = F.interpolate(imgs, self.model.cond_image_size, mode="nearest")
                        self.validation_prompts = [""] * self.batch_size

                    self.accelerator.print(f"[S{steps - 1}]{proc_label}: Logging validation images")

                    saved_image = self.save_validation_images(
                        self.validation_prompts, steps - 1, cond_image=cond_image
                    )
                    self.accelerator.print(f"[S{steps - 1}]{proc_label}: saved to {saved_image}")

                    self.counter_1 = 0


                if self.use_profiling:
                    counter += 1
                    if counter == self.profile_frequency:
                        # in order to use export_chrome_trace we need to first stop the profiler
                        prof.__exit__(None, None, None)
                        # show the information on the console using loguru as it provides better formating and we can later add colors for easy reading.
                        from loguru import logger
                        logger.info(prof.key_averages().table(sort_by='cpu_time_total', row_limit=self.row_limit))
                        # save the trace.json file with the information we gathered during this training step,
                        # we can use this trace.json file on the chrome tracing page or other similar tool to view more information.
                        prof.export_chrome_trace(f'{self.logging_dir}/trace.json')
                        # then we can restart it to continue reusing the same profiler.
                        prof = torch.autograd.profiler.profile(use_cuda=True)
                        prof.__enter__()
                        counter = 1 # Reset step counter

                if met is not None and not (steps % self.log_metrics_every):
                    self.accelerator.print(f"[S{steps - 1}]{proc_label}: metrics:")

                self.steps += (self.batch_size * self.gradient_accumulation_steps)

        # loop complete, save final model
        self.accelerator.print(f"[S{steps - 1}]{proc_label}[FINAL]: saving model to {self.results_dir}")
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        maskgit_save_name = "maskgit_superres" if self.model.cond_image_size else "maskgit"
        file_name = (
            f"{maskgit_save_name}.{steps - 1}.pt"
            if not self.only_save_last_checkpoint
            else f"{maskgit_save_name}.pt"
        )

        model_path = self.results_dir.joinpath(file_name)
        self.accelerator.wait_for_everyone()
        self.accelerator.save(state_dict, model_path)

        if self.args and not self.args.do_not_save_config:
            # save config file next to the model file.
            conf = OmegaConf.create(vars(self.args))
            OmegaConf.save(conf, f"{model_path}.yaml")

        if self.use_ema:
            self.accelerator.print(
                f"[S{steps - 1}]{proc_label}[FINAL]: saving EMA model to {self.results_dir}"
            )
            ema_state_dict = self.accelerator.unwrap_model(self.ema_model).state_dict()
            file_name = (
                f"{maskgit_save_name}.{steps - 1}.ema.pt"
                if not self.only_save_last_checkpoint
                else f"{maskgit_save_name}.ema.pt"
            )
            model_path = str(self.results_dir / file_name)
            self.accelerator.wait_for_everyone()
            self.accelerator.save(ema_state_dict, model_path)

            if self.args and not self.args.do_not_save_config:
                # save config file next to the model file.
                conf = OmegaConf.create(vars(self.args))
                OmegaConf.save(conf, f"{model_path}.yaml")

        cond_image = None
        if self.model.cond_image_size:
            self.accelerator.print(
                "With conditional image training, we recommend keeping the validation prompts to empty strings"
            )
            cond_image = F.interpolate(imgs[0], 256)
        steps = int(self.steps.item()) + 1  # get the final step count, plus one
        self.accelerator.print(f"[S{steps -1}]{proc_label}: Logging validation images")
        saved_image = self.save_validation_images(self.validation_prompts, steps - 1, cond_image=cond_image)
        self.accelerator.print(f"[S{steps - 1}]{proc_label}: saved to {saved_image}")

        if met is not None and not (steps % self.log_metrics_every):
            self.accelerator.print(f"[S{steps - 1}]{proc_label}: metrics:")
