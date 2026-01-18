# -*- coding: utf-8 -*-
import os, math, random, logging, json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- Core deps ----------------
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from einops import rearrange

# ---------------- Logging/accel ----------------
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    set_seed,
)

# ---------------- Diffusers/Transformers ----------------
import transformers
import diffusers
from yaml import load, Loader
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

# ---------------- Your utils (keep as-is) ----------------
from utils import init_logging, import_custom_class, save_video
from utils.model_utils import (
    load_condition_models, load_latent_models, load_vae_models, load_diffusion_model,
    count_model_parameters, unwrap_model
)
from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory
from utils.data_utils import (
    get_latents, get_text_conditions, gen_noise_from_condition_frame_latent,
    randn_tensor, apply_color_jitter_to_video
)
from utils.extra_utils import act_metric

# ---------------- NavSim deps ----------------
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

# ---------------- Config constants ----------------
CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"

logger = get_logger("unified_trainer")
LOG_LEVEL = "INFO"
logger.setLevel(LOG_LEVEL)


# ===========================
# Prompt builder (按你规则)
# ===========================
NAV_CMDS = ['turn left', 'go straight', 'turn right']

def one_hot_to_cmd(one_hot) -> str:
    if torch.is_tensor(one_hot):
        one_hot = one_hot.detach().cpu().tolist()
    elif isinstance(one_hot, np.ndarray):
        one_hot = one_hot.tolist()
    else:
        one_hot = list(one_hot)
    return next((NAV_CMDS[i] for i, v in enumerate(one_hot) if v == 1), "unknown")

def build_prompt_fixed(hist_xyh: torch.Tensor,
                       high_cmd_one_hot,
                       speed_mps: float,
                       acc_mps2: float) -> str:
    """
    hist_xyh: (T_hist, 3)  最后一帧 ~ (0,0,0)，第一帧为最早历史
    total_* / yaw 采用第一帧绝对值（你的要求）
    """
    h = hist_xyh.detach().cpu()
    x0, y0, th0 = h[0].tolist()

    total_forward  = abs(float(x0))
    total_lateral  = abs(float(y0))
    net_yaw_change = abs(float(th0))  # rad

    # 速度档位
    if speed_mps < 5.0:
        speed_desc = "at low speed"
    elif speed_mps < 15.0:
        speed_desc = "at moderate speed"
    else:
        speed_desc = "at highway speed"

    # 稳定性（加速度）
    stability_desc = "steady motion" if acc_mps2 < 0.5 else "gradually changing speed"

    # 指令 -> 文案
    cmd = one_hot_to_cmd(high_cmd_one_hot).lower()
    if "left" in cmd:
        motion_trend, turning_desc = "turning left", "with controlled steering"
    elif "right" in cmd:
        motion_trend, turning_desc = "turning right", "with controlled steering"
    elif "straight" in cmd:
        motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"
    else:
        motion_trend, turning_desc = "driving straight ahead", "with stable lane keeping"

    past_seconds, future_seconds = 2.0, 4.0
    prompt = (
        f"A high-quality, photorealistic dashboard camera view of autonomous driving. "
        f"Based on the past {past_seconds:.0f} seconds showing {motion_trend} {turning_desc}, "
        f"predict and generate the next {future_seconds:.0f} seconds of realistic driving continuation, "
        f"moving {speed_desc} with {stability_desc}. "
        f"Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
        f"clear details, and realistic physics. "
        f"[Technical: forward {total_forward:.2f}m, lateral {total_lateral:.2f}m, "
        f"yaw {np.degrees(net_yaw_change):.1f}°, speed {float(speed_mps):.2f}m/s]"
    )
    return prompt


# =========================================
# collate：NavSim -> 训练循环所需的 batch
# =========================================
def navsim_genie_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], dict]]
):
    """
    期望 features 至少包含：
      - images: (T,C,H,W) in [-1,1]
      - history_trajectory: (T_hist,3)
      - high_command_one_hot: (3,)
      - status_feature: [...], 约定 status_feature[0]=speed(m/s), [1]=acc(m/s^2)
      - last_hidden_state: (L,D)  可选
    期望 targets 至少包含：
      - trajectory: (L, action_dim)  —— 用作动作监督（若 return_action）
    """
    features_list, targets_list, _ = zip(*batch)

    # video -> (B,C,V=1,T,H,W)
    imgs = [f['images'].cpu() for f in features_list]  # each (T,C,H,W)
    video = torch.stack(imgs, dim=0).permute(0,2,1,3,4).unsqueeze(2).contiguous()

    # captions
    captions = []
    for f in features_list:
        hist = f['history_trajectory']
        cmd  = f['high_command_one_hot']
        if 'status_feature' in f:
            sf = f['status_feature'].reshape(-1)
            spd = float(sf[0].item()) if sf.numel() > 0 else 0.0
            acc = float(sf[1].item()) if sf.numel() > 1 else 0.0
        else:
            spd, acc = 0.0, 0.0
        captions.append(build_prompt_fixed(hist, cmd, spd, acc))

    # actions（作为监督）
    actions = torch.stack([t['trajectory'].cpu() for t in targets_list], dim=0)  # (B,L,C_action)


    out = {
        "video": video,          # (B,C,1,T,H,W)
        "caption": captions,     # list[str], len=B
        "actions": actions,      # (B,L,Ca)
    }

    return out


# ===========================
# 统一 Trainer（重写版）
# ===========================
class State:
    seed: int = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None
    output_dir: str = None


class UnifiedTrainer:
    """
    - 数据：来自 NavSim（prepare_dataset/prepare_val_dataset 内构建 SceneLoader/Dataset）
    - 模型/训练循环：沿用你之前的扩散训练逻辑（视频/动作分支、flow matching、accelerate）
    - 文本：caption 由 collate 用固定模板生成
    """

    def __init__(self, cfg,config_file):
        """
        cfg：你的 hydra 训练配置，需包含：
          - navsim 路径与 split：
              navsim_log_path, sensor_blobs_path, train_logs, val_logs,
              train_test_split.scene_filter, cache_path, force_cache_computation
          - dataloader.params: {batch_size, num_workers, ...}
          - genie_config: 模型/优化相关 yaml 路径（沿用你的 genie 配置）
          - output_dir, logging_dir, mixed_precision, gradient_accumulation_steps, etc.
        """
        self.cfg = cfg

        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        self.args = args

        self.state = State()
        # self.tokenizer = None
        # self.text_encoder = None
        # self.diffusion_model = None
        # self.vae = None
        # self.scheduler = None
        # self.pipeline_class = None
        self.agent = instantiate(cfg.agent)

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        # 日志 & 保存目录
        if self.state.accelerator.is_main_process:
            start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.save_folder = os.path.join(self.cfg.output_dir, start_time)
            os.makedirs(self.save_folder, exist_ok=True)
            with open(os.path.join(self.save_folder, 'navsim_config.json'), "w") as f:
                json.dump(json.loads(str(self.cfg)), f, indent=2)
        else:
            self.save_folder = self.cfg.output_dir

    # -------- 初始化 ----------
    def _init_distributed(self):
        logging_dir = Path(self.cfg.output_dir, self.cfg.logging_dir)
        project_config = ProjectConfiguration(project_dir=self.cfg.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_pg_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout))
        mixed_precision = "no" if torch.backends.mps.is_available() else self.cfg.mixed_precision
        report_to = None if str(getattr(self.cfg, "report_to", "none")).lower() == "none" else self.cfg.report_to

        ds_plugin = None
        if getattr(self.cfg, "use_deepspeed", False):
            per_device_bs = self.cfg.dataloader.params.batch_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            grad_accum = self.cfg.gradient_accumulation_steps
            train_batch_size = per_device_bs * world_size * grad_accum
            self.args.deepspeed["train_batch_size"] = train_batch_size
            ds_plugin = DeepSpeedPlugin(hf_ds_config=self.args.deepspeed,
                                        gradient_accumulation_steps=grad_accum)

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
            deepspeed_plugin=ds_plugin,
        )
        if torch.backends.mps.is_available():
            accelerator.native_amp = False
        self.state.accelerator = accelerator

        if getattr(self.cfg, "seed", None) is not None:
            self.state.seed = self.cfg.seed
            set_seed(self.cfg.seed)

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        self.state.weight_dtype = weight_dtype

    def _init_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.state.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        logger.info("Initialized UnifiedTrainer")
        logger.info(self.state.accelerator.state, main_process_only=False)

    def _init_directories(self):
        if self.state.accelerator.is_main_process:
            Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
            self.state.output_dir = self.cfg.output_dir

    # -------- 数据集构建 ----------
    def prepare_dataset(self):
        logger.info("Building NavSim training dataset")
        #agent: AbstractAgent = instantiate(self.cfg.agent)
        if self.cfg.use_cache_without_dataset:
            self.train_dataset = CacheOnlyDataset(
                cache_path=self.cfg.cache_path,
                feature_builders=self.agent.get_feature_builders(),
                target_builders=self.agent.get_target_builders(),
                log_names=self.cfg.train_logs,
            )
        else:
            train_scene_filter: SceneFilter = instantiate(self.cfg.train_test_split.scene_filter)
            train_scene_filter.log_names = self.cfg.train_logs if train_scene_filter.log_names is None else [
                n for n in train_scene_filter.log_names if n in self.cfg.train_logs
            ]

            data_path = Path(self.cfg.navsim_log_path)
            sensor_blobs_path = Path(self.cfg.sensor_blobs_path)
            train_scene_loader = SceneLoader(
                sensor_blobs_path=sensor_blobs_path,
                data_path=data_path,
                scene_filter=train_scene_filter,
                sensor_config=self.agent.get_sensor_config(),
            )

            self.train_dataset = Dataset(
                scene_loader=train_scene_loader,
                feature_builders=self.agent.get_feature_builders(),
                target_builders=self.agent.get_target_builders(),
                cache_path=self.cfg.cache_path,
                force_cache_computation=self.cfg.force_cache_computation,
            )

        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataloader.params.batch_size,
            num_workers=self.cfg.dataloader.params.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=navsim_genie_collate_fn,
        )
        logger.info(f"Train samples: {len(self.train_dataset)}")

    def prepare_val_dataset(self):
        logger.info("Building NavSim validation dataset")
        #agent: AbstractAgent = instantiate(self.cfg.agent)
        if self.cfg.use_cache_without_dataset:
            self.val_dataset = CacheOnlyDataset(
                cache_path=self.cfg.cache_path,
                feature_builders=self.agent.get_feature_builders(),
                target_builders=self.agent.get_target_builders(),
                log_names=self.cfg.val_logs,
            )
        else:
            val_scene_filter: SceneFilter = instantiate(self.cfg.train_test_split.scene_filter)
            val_scene_filter.log_names = self.cfg.val_logs if val_scene_filter.log_names is None else [
                n for n in val_scene_filter.log_names if n in self.cfg.val_logs
            ]

            data_path = Path(self.cfg.navsim_log_path)
            sensor_blobs_path = Path(self.cfg.sensor_blobs_path)
            val_scene_loader = SceneLoader(
                sensor_blobs_path=sensor_blobs_path,
                data_path=data_path,
                scene_filter=val_scene_filter,
                sensor_config=self.agent.get_sensor_config(),
            )

            self.val_dataset = Dataset(
                scene_loader=val_scene_loader,
                feature_builders=self.agent.get_feature_builders(),
                target_builders=self.agent.get_target_builders(),
                cache_path=self.cfg.cache_path,
                force_cache_computation=self.cfg.force_cache_computation,
            )

        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.cfg.dataloader.params.batch_size,
            num_workers=self.cfg.dataloader.params.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=navsim_genie_collate_fn,
        )
        logger.info(f"Val samples: {len(self.val_dataset)}")

    # -------- 模型/优化器等 ----------
    def prepare_models(self):
        logger.info("Initializing models")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype

        self.agent.initialize()

        self.tokenizer       = self.agent.tokenizer
        self.text_encoder    = self.agent.text_encoder
        self.vae             = self.agent.vae
        self.diffusion_model = self.agent.diffusion_model
        self.scheduler       = self.agent.scheduler
        self.pipeline_class  = self.agent.pipeline_class
        
        text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="")
        self.uncond_prompt_embeds = text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = text_uncond['prompt_attention_mask']

        self.SPATIAL_DOWN_RATIO  = getattr(self.agent, "SPATIAL_DOWN_RATIO", self.vae.spatial_compression_ratio)
        self.TEMPORAL_DOWN_RATIO = getattr(self.agent, "TEMPORAL_DOWN_RATIO", self.vae.temporal_compression_ratio)


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        if self.cfg.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        if self.args.gradient_checkpointing:
            self.diffusion_model.enable_gradient_checkpointing()

        if self.cfg.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

        train_mode = self.args.train_mode
        trainable_params = []
        for name, p in self.diffusion_model.named_parameters():
            if train_mode == 'action_only':
                p.requires_grad = ('action_' in name)
            elif train_mode == "video_only":
                p.requires_grad = ('action_' not in name)
            elif train_mode in ("all", "action_full"):
                p.requires_grad = True
            else:
                raise NotImplementedError
            if p.requires_grad:
                trainable_params.append(p)

        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable params: {self.state.num_trainable_parameters}")

        self.state.learning_rate = self.args.lr
        if self.cfg.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.cfg.gradient_accumulation_steps
                * self.cfg.dataloader.params.batch_size
                * self.state.accelerator.num_processes
            )

        params_to_optimize = [{"params": trainable_params, "lr": self.state.learning_rate}]
        self.optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit=getattr(self.args, "optimizer_8bit", False),
            use_torchao=getattr(self.args, "optimizer_torchao", False),
        )

        # steps
        num_upd_per_epoch = math.ceil(len(self.train_dataloader) / self.cfg.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.args.train_steps or (self.args.train_epochs * num_upd_per_epoch)
            self.state.overwrote_max_train_steps = True
        self.state.train_epochs = self.args.train_epochs

        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

    def prepare_for_training(self):
        self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.state.accelerator.prepare(self.diffusion_model, self.optimizer, self.train_dataloader, self.lr_scheduler)

    def prepare_trackers(self):
        self.state.accelerator.init_trackers(self.args.tracker_name or "navsim_train", config=self.args.__dict__)

    # -------- 训练 ----------
    def train(self):
        logger.info("Starting training")
        logger.info("Memory before: %s", json.dumps(get_memory_statistics(), indent=2))

        accel = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accel.device, dtype=weight_dtype)
        generator = torch.Generator(device=accel.device)
        if self.cfg.seed is not None:
            generator = generator.manual_seed(self.cfg.seed)
        self.state.generator = generator

        progress = range(self.state.train_steps)
        global_step = 0
        running_loss = 0.0

        for epoch in range(self.state.train_epochs):
            self.diffusion_model.train()
            for step, batch in enumerate(self.train_dataloader):
                logs = {}
                with accel.accumulate([self.diffusion_model]):
                    video = batch['video'].to(accel.device, dtype=weight_dtype)  # (B,C,1,T,H,W)
                    b, c, v, t, h, w = video.shape
                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')
                    mem = video                       
                    mem_size = mem.shape[2] 

                    if getattr(self.args, "use_color_jitter", False):
                        mem = apply_color_jitter_to_video(mem)


                    # 文本条件（带 dropout）
                    captions = batch['caption']
                    dropout_factor = torch.rand(b, device=accel.device, dtype=weight_dtype)
                    dropout_mask_prompt = (dropout_factor < self.args.caption_dropout_p).unsqueeze(1).unsqueeze(2)
                    text_conds = get_text_conditions(self.tokenizer, self.text_encoder, captions)
                    prompt_embeds = text_conds['prompt_embeds']
                    prompt_attention_mask = text_conds['prompt_attention_mask']
                    prompt_embeds = self.uncond_prompt_embeds.repeat(b,1,1)*dropout_mask_prompt + \
                                    prompt_embeds*~dropout_mask_prompt

                    # 潜空间 + 条件掩码
                    mem_latents, _ = get_latents(self.vae, mem, mem[:, :, :0])

                    latent_height = h // self.SPATIAL_DOWN_RATIO
                    latent_width  = w // self.SPATIAL_DOWN_RATIO
                    mem_latents = rearrange(mem_latents, '(b v m) (hh ww) ch -> (b v) ch m hh ww',
                                            b=b, v=v, m=mem_size, hh=latent_height, ww=latent_width)

                    # 现在的 5D latent 只有历史帧
                    latents_5d = mem_latents                      # (B*V, C, M, H', W')
                    latent_frames = mem_size                      # 关键：只等于历史帧数

                    # 生成噪声/时步/掩码（把历史帧视为条件，屏蔽其损失；我们也不算视频损失）
                    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                        mem_latents, latent_frames, latent_height, latent_width,
                        noise_to_condition_frames=self.args.noise_to_first_frame
                    )
                    latents = rearrange(latents_5d, 'bv ch f hh ww -> bv (f hh ww) ch')  # (B*V, THW, C)

                    # flow-matching 时步/噪声（视频）
                    weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=b,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    weights  = rearrange(weights.unsqueeze(1).repeat(1, v), 'b v -> (b v)')
                    indices  = (weights * self.scheduler.config.num_train_timesteps).long()
                    sigmas   = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()
                    if self.args.pixel_wise_timestep:
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)   # (B*V, THW)
                    else:
                        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)       # (B*V, T)

                    ss = sigmas.reshape(-1,1,1).repeat(1,1,latents.size(-1))
                    if getattr(self.args, "noisy_video", False):
                        ss = torch.full_like(ss, 1.0)   # 也可以把视频侧完全当噪声处理
                    noisy = randn_tensor(latents.shape, device=accel.device, dtype=weight_dtype)
                    noisy_latents = (1.0 - ss) * latents + ss * noisy

                    actions = batch['actions'].to(accel.device, dtype=weight_dtype)  # (B,L,Ca)
                    action_dim = actions.shape[-1]
                    noise_actions = randn_tensor(actions.shape, device=accel.device, dtype=weight_dtype)

                    action_weights = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=b,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )
                    action_indices   = (action_weights * self.scheduler.config.num_train_timesteps).long()
                    action_sigmas    = scheduler_sigmas[action_indices]
                    action_timesteps = (action_sigmas * 1000.0).long().unsqueeze(-1).repeat(1, actions.shape[1])
                    action_ss        = action_sigmas.reshape(-1,1,1).repeat(1,1,action_dim)
                    noisy_actions    = (1.0 - action_ss) * actions + action_ss * noise_actions

                    act_state = batch.get('state', None)
                    if act_state is not None:
                        act_state = act_state.to(accel.device, dtype=weight_dtype)

                    # 这里让视频分支运行（用历史帧 tokens），仅为了产生 video states 供动作头用；但不计算视频损失
                    pred_all = self._forward_pass(
                        latents=noisy_latents,
                        timesteps=timesteps,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        latent_frames=latent_frames,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        n_view=v,
                        return_video=True,                  # 运行视频侧得到 video states
                        return_action=True,
                        video_attention_mask=None,
                        history_action_state=act_state,
                        conditioning_mask=conditioning_mask,
                        action_timestep=action_timesteps,
                        action_states=noisy_actions,
                        action_dim=action_dim
                    )
                    pred_action = pred_all['action']        # (B, L, Ca)

                    action_w = compute_loss_weighting_for_sd3(self.args.flow_weighting_scheme, action_sigmas)
                    action_w = action_w.reshape(-1,1,1).repeat(1,1,action_dim)
                    target_action = noise_actions - actions
                    loss_action = (action_w.float() * (pred_action.float() - target_action.float()).pow(2)).mean()
                    loss = loss_action     

                    assert not torch.isnan(loss), "NaN loss detected"
                    accel.backward(loss)
                    if accel.sync_gradients and accel.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accel.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # logging
                loss = accel.reduce(loss.detach(), reduction='mean')
                if self.args.train_mode in ('all','action_only','action_full'):
                    loss_action = accel.reduce(loss_action.detach(), reduction='mean')
                if self.args.train_mode in ('all','video_only'):
                    loss_video = accel.reduce(loss_video.detach(), reduction='mean')

                running_loss += loss.item()
                if accel.sync_gradients:
                    global_step += 1

                logs.update({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]})
                accel.log(logs, step=global_step)

                # 验证
                if global_step % self.args.steps_to_val == 0 and self.val_dataloader is not None and accel.is_main_process:
                    model_save_dir = os.path.join(self.save_folder, f'Validation_step_{global_step}')
                    self.validate(accel, model_save_dir, global_step, n_view=v, n_chunk=1)

                # 保存
                if global_step % self.args.steps_to_save == 0 and accel.is_main_process:
                    model_to_save = unwrap_model(accel, self.diffusion_model)
                    model_save_dir = os.path.join(self.save_folder, f'step_{global_step}')
                    model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                    del model_to_save

                if global_step >= self.state.train_steps:
                    logger.info("Reached max train steps")
                    break

            logger.info(f"Epoch {epoch+1} done. Mem: {json.dumps(get_memory_statistics(), indent=2)}")

        accel.wait_for_everyone()
        if accel.is_main_process:
            self.diffusion_model = unwrap_model(accel, self.diffusion_model)
            model_save_dir = os.path.join(self.save_folder, f'step_{global_step}')
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

        del self.diffusion_model, self.scheduler
        free_memory()
        logger.info(f"Memory after training: {json.dumps(get_memory_statistics(), indent=2)}")
        accel.end_training()

    def _forward_pass(self, latents, timesteps, prompt_embeds, prompt_attention_mask,
                      latent_frames, latent_height, latent_width, n_view,
                      return_video, return_action, video_attention_mask,
                      history_action_state, conditioning_mask,
                      action_timestep=None, action_states=None, action_dim=None):
        """薄封装，兼容你 utils.forward_pass 的接口/返回。"""
        return forward_pass(
            model=self.diffusion_model,
            timesteps=timesteps,
            noisy_latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            n_view=n_view,
            action_states=action_states,
            action_timestep=action_timestep,
            return_video=return_video,
            return_action=return_action,
            video_attention_mask=video_attention_mask,
            history_action_state=history_action_state,
            condition_mask=conditioning_mask,
        )['latents']

    # -------- 验证 ----------
    @torch.inference_mode()
    def validate(self, accelerator, model_save_dir, global_step, n_view=1, n_chunk=30):
        os.makedirs(model_save_dir, exist_ok=True)
        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer,
            unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model
        )

        batch = next(iter(self.val_dataloader))
        image = batch['video'][:, :, :, :self.args.data['train']['n_previous']]  # (B,C,V,T,H,W)
        prompt = batch['caption']
        gt_video = batch['video']

        b, c, v, t, h, w = image.shape
        image = rearrange(image, 'b c v t h w -> (b v) c t h w')
        num_steps = self.args.num_inference_step

        history_action_state = batch.get('state', None)

        preds = pipe.infer(
            image=image,
            prompt=prompt[:1],  # 取一个样本可视化
            negative_prompt='',
            num_inference_steps=num_steps,
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            return_action=self.args.return_action,
            n_prev=self.args.data['train']['n_previous'],
            chunk=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1,
            return_video=self.args.return_video,
            noise_seed=42,
            action_chunk=self.args.data['train']['action_chunk'],
            history_action_state=history_action_state,
            pixel_wise_timestep=self.args.pixel_wise_timestep,
            n_chunk=n_chunk,
            action_dim=self.args.diffusion_model["config"]["action_in_channels"],
        )[0]

        # 保存 GT 与 生成
        save_video(rearrange(gt_video[0].data.cpu(), 'c v t h w -> c t h (v w)', v=v),
                   os.path.join(model_save_dir, 'val_gt.mp4'),
                   fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)
        if self.args.return_video:
            video = preds['video'].data.cpu()
            save_video(rearrange(video, '(b v) c t h w -> b c t h (v w)', v=v)[0],
                       os.path.join(model_save_dir, 'val_pred.mp4'),
                       fps=(self.args.data['train']['chunk']-1)//self.TEMPORAL_DOWN_RATIO+1)

        if self.args.return_action:
            gt_actions = batch['actions'][:, -self.args.data['train']['action_chunk']:]
            action_dim = gt_actions.shape[-1]
            logs = act_metric(
                preds['action'][:, :, :action_dim].detach().cpu().float().numpy()[:1],
                gt_actions[:, :, :action_dim].detach().cpu().float().numpy()[:1],
                prefix='val',
                start_stop_interval=[(0,1),(1,9),(9,25),(25,self.args.data['train']['action_chunk'])]
            )
            for k,v in logs.items():
                accelerator.log({k: v}, step=global_step)


# ===========================
# Hydra main
# ===========================
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    trainer = UnifiedTrainer(cfg,'/mnt/evad_fs/worldmodel/yongkangli/navsim-1.1/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml')
    trainer.prepare_dataset()
    trainer.prepare_val_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_for_training()
    trainer.prepare_trackers()
    trainer.train()

if __name__ == "__main__":
    main()
