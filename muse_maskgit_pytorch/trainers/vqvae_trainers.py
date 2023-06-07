from PIL import Image
from torchvision.utils import make_grid, save_image
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from einops import rearrange
from ema_pytorch import EMA
import numpy as np
import torch
from accelerate import Accelerator
from muse_maskgit_pytorch.trainers.base_accelerated_trainer import (
    BaseAcceleratedTrainer,
    get_optimizer,
)
from diffusers.optimization import get_scheduler
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from omegaconf import OmegaConf

def noop(*args, **kwargs):
    pass


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.0)
        log[key] = old_value + new_value
    return log


def exists(val):
    return val is not None


class VQGanVAETrainer(BaseAcceleratedTrainer):
    def __init__(
        self,
        vae: VQGanVAE,
        dataloader: DataLoader,
        valid_dataloader: DataLoader,
        accelerator: Accelerator,
        *,
        current_step,
        num_train_steps,
        batch_size,
        gradient_accumulation_steps=1,
        max_grad_norm=None,
        save_results_every=100,
        save_model_every=1000,
        results_dir="./results",
        project_dir="./results/logs",
        apply_grad_penalty_every=4,
        lr=3e-4,
        lr_scheduler_type="constant",
        lr_warmup_steps=500,
        num_cycles=1,
        scheduler_power=1.0,
        discr_max_grad_norm=None,
        use_ema=True,
        ema_beta=0.995,
        ema_update_after_step=0,
        ema_update_every=1,
        clear_previous_experiments=False,
        validation_image_scale: float = 1.0,
        only_save_last_checkpoint=False,
        use_profiling=False,
        profile_frequency=1,
        row_limit=10,
        optimizer="Adam",
        weight_decay=0.0,
        use_8bit_adam=False,
        args=None,
    ):
        super().__init__(
            dataloader,
            valid_dataloader,
            accelerator,
            current_step=current_step,
            num_train_steps=num_train_steps,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_results_every=save_results_every,
            save_model_every=save_model_every,
            results_dir=results_dir,
            project_dir=project_dir,
            apply_grad_penalty_every=apply_grad_penalty_every,
            clear_previous_experiments=clear_previous_experiments,
            validation_image_scale=validation_image_scale,
            only_save_last_checkpoint=only_save_last_checkpoint,
            use_profiling=use_profiling,
            profile_frequency=profile_frequency,
            row_limit=row_limit,
        )

        self.current_step = current_step
        self.counter_1 = 0
        self.counter_2 = 0

        # arguments used for the training script,
        # we are going to use them later to save them to a config file.
        self.args = args

        # vae
        self.model = vae

        all_parameters = set(vae.parameters())
        discr_parameters = set(vae.discr.parameters())
        vae_parameters = all_parameters - discr_parameters

        # optimizers
        self.optim = get_optimizer(use_8bit_adam, optimizer, vae_parameters, lr, weight_decay)
        self.discr_optim = get_optimizer(use_8bit_adam, optimizer, discr_parameters, lr, weight_decay)

        self.lr_scheduler: LRScheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.optim,
            num_warmup_steps=lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.num_train_steps * self.gradient_accumulation_steps,
            num_cycles=num_cycles,
            power=scheduler_power,
        )

        self.lr_scheduler_discr: LRScheduler = get_scheduler(
            lr_scheduler_type,
            optimizer=self.discr_optim,
            num_warmup_steps=lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.num_train_steps * self.gradient_accumulation_steps,
            num_cycles=num_cycles,
            power=scheduler_power,
        )

        self.discr_max_grad_norm = discr_max_grad_norm

        # prepare with accelerator

        (
            self.model,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl,
            self.lr_scheduler,
            self.lr_scheduler_discr,
        ) = accelerator.prepare(
            self.model,
            self.optim,
            self.discr_optim,
            self.dl,
            self.valid_dl,
            self.lr_scheduler,
            self.lr_scheduler_discr,
        )
        self.model.train()

        self.use_ema = use_ema

        if use_ema:
            self.ema_model = EMA(
                vae,
                update_after_step=ema_update_after_step,
                update_every=ema_update_every,
            )
            self.ema_model = accelerator.prepare(self.ema_model)

    def load(self, path):
        pkg = super().load(path)
        self.discr_optim.load_state_dict(pkg["discr_optim"])

    def save(self, path):
        if not self.is_local_main_process:
            return

        pkg = dict(
            model=self.get_state_dict(self.model),
            optim=self.optim.state_dict(),
            discr_optim=self.discr_optim.state_dict(),
        )
        self.accelerator.save(pkg, path)

    def log_validation_images(self, models_to_evaluate, logs, steps):
        log_imgs = []
        prompts = ["vae"] if len(models_to_evaluate) == 1 else ["vae", "ema"]
        for model, filename in models_to_evaluate:
            model.eval()

            valid_data = next(self.valid_dl_iter)
            valid_data = valid_data.to(self.device)

            recons = model(valid_data, return_recons=True)

            # else save a grid of images
            imgs_and_recons = torch.stack((valid_data, recons), dim=0)
            imgs_and_recons = rearrange(imgs_and_recons, "r b ... -> (b r) ...")

            imgs_and_recons = imgs_and_recons.detach().cpu().float().clamp(0.0, 1.0)
            grid = make_grid(
                imgs_and_recons, nrow=2, normalize=True, value_range=(0, 1)
            )

            # Fix aspect ratio and scale the image size if needed
            if self.validation_image_scale != 1:
                img_size = grid.shape[-2:]
                output_size = (
                    int(img_size[0] * self.validation_image_scale),
                    int(img_size[1] * self.validation_image_scale),
                )
                grid = TF.resize(grid, output_size)

            logs["reconstructions"] = grid
            save_file = str(self.results_dir / f"{filename}.png")
            save_image(grid, save_file)
            log_imgs.append(np.asarray(Image.open(save_file)))

        super().log_validation_images(log_imgs, steps, prompts=prompts)

    def train(self):
        self.steps = self.steps + 1
        device = self.device

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

        while int(self.steps.item()) < self.num_train_steps:
            # update the tqdm progress bar
            pbar.update(self.batch_size * self.gradient_accumulation_steps)

            try:
                for img in self.dl:
                    steps = int(self.steps.item())

                    apply_grad_penalty = (steps % self.apply_grad_penalty_every) == 0

                    self.model.train()
                    discr = self.model.module.discr if self.is_distributed else self.model.discr
                    if self.use_ema:
                        ema_model = self.ema_model.module if self.is_distributed else self.ema_model

                    # logs

                    logs = {}

                    # update vae (generator)

                    for _ in range(self.gradient_accumulation_steps):
                        img = next(self.dl_iter)
                        img = img.to(device)

                        with self.accelerator.autocast():
                            loss = self.model(
                                img, add_gradient_penalty=apply_grad_penalty, return_loss=True
                            )

                        self.accelerator.backward(loss / self.gradient_accumulation_steps)

                        accum_log(
                            logs, {"Train/vae_loss": loss.item() / self.gradient_accumulation_steps}
                        )

                    if exists(self.max_grad_norm):
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                    self.lr_scheduler.step()
                    self.lr_scheduler_discr.step()
                    self.optim.step()
                    self.optim.zero_grad()

                    # update discriminator

                    if exists(discr):
                        self.discr_optim.zero_grad()

                        for _ in range(self.gradient_accumulation_steps):
                            img = next(self.dl_iter)
                            img = img.to(device)

                            #with torch.cuda.amp.autocast():
                        loss = self.model(img, return_discr_loss=True)

                    self.accelerator.backward(loss / self.gradient_accumulation_steps)
                    if self.discr_max_grad_norm is not None and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    accum_log(
                            logs,
                            {"Train/discr_loss": loss.item() / self.gradient_accumulation_steps},
                        )

                    if exists(self.discr_max_grad_norm):
                        self.accelerator.clip_grad_norm_(
                                        discr.parameters(), self.discr_max_grad_norm
                                    )

                    self.discr_optim.step()

                    # log

                    self.accelerator.print(
                            f"{steps}: vae loss: {logs['Train/vae_loss']} - discr loss: {logs['Train/discr_loss']} - lr: {self.lr_scheduler.get_last_lr()[0]}"
                        )
                    logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                    self.accelerator.log(logs, step=steps)

                    # we made two counters, one for the results and one for the
                    # model so we can properly save them without any error
                    # no matter what batch_size and gradient_accumulation_steps we use.
                    self.counter_1 += (self.batch_size * self.gradient_accumulation_steps)
                    self.counter_2 += (self.batch_size * self.gradient_accumulation_steps)

                    # update exponential moving averaged generator

                    if self.use_ema:
                        ema_model.update()

                    # sample results every so often
                    logs['save_results_every'] = ''
                    if self.counter_1 == self.save_results_every:
                        vaes_to_evaluate = ((self.model, str(steps - 1)),)

                        if self.use_ema:
                            vaes_to_evaluate = ((ema_model.ema_model, f"{steps - 1}.ema"),) + vaes_to_evaluate

                        self.log_validation_images(vaes_to_evaluate, logs, steps - 1)
                        #self.accelerator.print(f"{steps}: saving to {str(self.results_dir)}")
                        info_bar.set_description_str(f"VAE loss: {logs['Train/vae_loss']} - discr loss: {logs['Train/discr_loss']} - lr: {logs['lr']}")
                        self.accelerator.print(f"\nStep: {steps - 1} | Saving to {str(self.results_dir)}")

                        self.counter_1 = 0

                    # save model every so often
                    logs['save_model_every'] = ''
                    self.accelerator.wait_for_everyone()
                    if steps != self.current_step:
                        if self.counter_2 == self.save_model_every or steps == self.num_train_steps:
                            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
                            file_name = f"vae.{steps}.pt" if not self.only_save_last_checkpoint else "vae.pt"
                            model_path = str(self.results_dir / file_name)
                            self.accelerator.save(state_dict, model_path)

                            if self.args and not self.args.do_not_save_config:
                                # save config file next to the model file.
                                conf = OmegaConf.create(vars(self.args))
                                OmegaConf.save(conf, f"{model_path}.yaml")

                        if self.use_ema:
                            ema_state_dict = self.accelerator.unwrap_model(
                                                self.ema_model
                                                ).state_dict()

                            file_name = (f"vae.{steps - 1}.ema.pt" if not self.only_save_last_checkpoint else "vae.ema.pt")
                            model_path = str(self.results_dir / file_name)
                            self.accelerator.save(ema_state_dict, model_path)

                            if self.args and not self.args.do_not_save_config:
                                # save config file next to the model file.
                                conf = OmegaConf.create(vars(self.args))
                                OmegaConf.save(conf, f"{model_path}.yaml")

                        self.counter_2 = 0

                        self.accelerator.print(f"{steps}: saving model to {str(self.results_dir)}")

                    self.steps = self.steps + (self.batch_size * self.gradient_accumulation_steps)


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
                            prof.export_chrome_trace(f'{self.project_dir}/trace.json')
                            # then we can restart it to continue reusing the same profiler.
                            prof = torch.autograd.profiler.profile(use_cuda=True)
                            prof.__enter__()
                            counter = 1 # Reset step counter

            except StopIteration:
                self.dl = iter(self.dl)
                self.accelerator.print("Reached end of dataset, reloading data")

        # Loop finished, save model
        self.accelerator.wait_for_everyone()
        if self.is_main_process:
            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            file_name = f"vae.{steps}.pt" if not self.only_save_last_checkpoint else "vae.pt"
            model_path = str(self.results_dir / file_name)
            self.accelerator.save(state_dict, model_path)

            if self.args and not self.args.do_not_save_config:
                # save config file next to the model file.
                conf = OmegaConf.create(vars(self.args))
                OmegaConf.save(conf, f"{model_path}.yaml")

            if self.use_ema:
                ema_state_dict = self.accelerator.unwrap_model(
                                    self.ema_model
                                    ).state_dict()

                file_name = (f"vae.{steps - 1}.ema.pt" if not self.only_save_last_checkpoint else "vae.ema.pt")
                model_path = str(self.results_dir / file_name)
                self.accelerator.save(ema_state_dict, model_path)

            if self.args and not self.args.do_not_save_config:
                # save config file next to the model file.
                conf = OmegaConf.create(vars(self.args))
                OmegaConf.save(conf, f"{model_path}.yaml")

            self.accelerator.print(f"{steps}: saving model to {str(self.results_dir)}")

            # close the progress bar as we no longer need it.
            pbar.close()
            self.accelerator.print("Training Complete.")