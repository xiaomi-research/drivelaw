# -*- coding: utf-8 -*-
import os, math, random, logging, json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Any, Union

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.functional as F

# ---------------- Core deps ----------------
import hydra
import argparse
from hydra.utils import instantiate
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from tqdm.auto import tqdm
from torch import Tensor

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
from diffusers.training_utils import EMAModel
from diffusers import AutoencoderKLLTXVideo

# ---------------- Your utils (keep as-is) ----------------
from utils import init_logging, import_custom_class, save_video
from utils.model_utils import (
    load_condition_models, load_latent_models, load_vae_models, load_diffusion_model,
    count_model_parameters, unwrap_model
)
from utils.model_utils import forward_pass

from utils.optimizer_utils import get_optimizer
from utils.memory_utils import get_memory_statistics, free_memory
from utils.data_utils import (
    get_latents, get_text_conditions, gen_noise_from_condition_frame_latent,
    randn_tensor, apply_color_jitter_to_video
)
from utils.timestep_samplers import SAMPLERS

from utils.extra_utils import act_metric
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition

# ---------------- NavSim deps ----------------
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset

# ---------------- Config constants ----------------
CONFIG_PATH = "navsim/planning/script/config/training"
CONFIG_NAME = "default_training"

logger = get_logger("unified_trainer")
LOG_LEVEL = "INFO"
logger.setLevel(LOG_LEVEL)


# ===========================
# Prompt builder
# ===========================
NAV_CMDS = ['turn left', 'go straight', 'turn right']

def _round_up_to(x: int, m: int) -> int:
    return (x + m - 1) // m * m

def _resize_to_multiple(img: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    if img.dim() == 3:  # (C,H,W)
        C,H,W = img.shape
        Hn, Wn = _round_up_to(H, multiple), _round_up_to(W, multiple)
        if (Hn, Wn) == (H, W): return img
        return F.interpolate(img.unsqueeze(0), size=(Hn, Wn), mode="bilinear", align_corners=False).squeeze(0)
    elif img.dim() == 4:  # (T,C,H,W)
        T,C,H,W = img.shape
        Hn, Wn = _round_up_to(H, multiple), _round_up_to(W, multiple)
        if (Hn, Wn) == (H, W): return img
        return F.interpolate(img, size=(Hn, Wn), mode="bilinear", align_corners=False)  
    else:
        raise ValueError(f"Unexpected img shape {img.shape}")

def _resize_to_hw(img: torch.Tensor, height: int = 704, width: int = 1280) -> torch.Tensor:
    """
    Supported shapes:
      - (C, H, W)
      - (T, C, H, W) where T is treated as a batch dimension.
    Always resizes to (height, width); returns input directly if it already has this size.
    """
    if img.dim() == 3:  # (C,H,W)
        _, H, W = img.shape
        if (H, W) == (height, width):
            return img
        return F.interpolate(
            img.unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(0)
    elif img.dim() == 4:  # (T,C,H,W)
        _, _, H, W = img.shape
        if (H, W) == (height, width):
            return img
        return F.interpolate(
            img, size=(height, width), mode="bilinear", align_corners=False
        )
    else:
        raise ValueError(f"Unexpected img shape {img.shape}")
        
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
    hist_xyh: (T_hist, 3), with the last frame approximately (0, 0, 0) and the first
        frame being the earliest history.
    total_* / yaw use the absolute value of the first frame.
    """
    h = hist_xyh.detach().cpu()
    x0, y0, th0 = h[0].tolist()

    total_forward  = abs(float(x0))
    total_lateral  = abs(float(y0))
    net_yaw_change = abs(float(th0))  # rad

    if speed_mps < 5.0:
        speed_desc = "at low speed"
    elif speed_mps < 15.0:
        speed_desc = "at moderate speed"
    else:
        speed_desc = "at highway speed"

    stability_desc = "steady motion" if acc_mps2 < 0.5 else "gradually changing speed"

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
        f"Based on the past {past_seconds:.0f} seconds videos, "
        f"predict and generate the next {future_seconds:.0f} seconds of realistic driving continuation, "
        f"Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
        f"clear details, and realistic physics. "
    )
    return prompt


def _to_chw_float_tensor(img_rgb_uint8: np.ndarray, normalize="[-1,1]") -> torch.Tensor:
    arr = img_rgb_uint8.astype(np.float32) / 255.0
    if normalize == "[-1,1]":
        arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C,H,W)




class CacheOnlyDatasetLazyIO(CacheOnlyDataset):
    """
    Cache-only dataset with lazy image loading.
      1) features["images"]: already a (T, C, H, W) tensor — returned as is.
      2) features["image_paths"]: List[str | Path] — lazily loaded to a tensor in __getitem__.
    Targets follow the same pattern:
      - targets["future_frames"]: (Tf, C, H, W) tensor.
      - targets["future_paths"]: List[str | Path], lazily loaded.
    """

    def __init__(
        self,
        cache_path: str,
        feature_builders: List["AbstractFeatureBuilder"],
        target_builders: List["AbstractTargetBuilder"],
        log_names: List[str] | None = None,
        *,
        normalize: str = "[-1,1]",
        min_hist_frames: int | None = None,   # Optional: minimum history length (e.g., 8).
        min_fut_frames: int | None = None,    # Optional: minimum future length if required.
    ):
        super().__init__(cache_path, feature_builders, target_builders, log_names)
        assert normalize in ("[-1,1]", "[0,1]")
        self.normalize = normalize
        self.min_hist_frames = min_hist_frames
        self.min_fut_frames = min_fut_frames

    def _load_one_image(self, p: Union[str, Path]) -> torch.Tensor:
        p = Path(p)
        # Read with PIL and convert to RGB.
        img = Image.open(p).convert("RGB")
        arr = np.array(img)  # (H,W,3) uint8
        return _to_chw_float_tensor(arr, normalize=self.normalize)

    def _maybe_load_images_from_paths(self, maybe_paths: Any) -> torch.Tensor:
        """
        Supported inputs:
          - Tensor: returned directly.
          - List[str | Path]: lazily loaded to a (T, C, H, W) tensor.
        """
        if isinstance(maybe_paths, torch.Tensor):
            return maybe_paths
        if isinstance(maybe_paths, (list, tuple)) and len(maybe_paths) > 0:
            imgs = [self._load_one_image(p) for p in maybe_paths]
            return torch.stack(imgs, dim=0)
        # Empty input: return an empty tensor as a placeholder.
        return torch.empty(0, 3, 0, 0)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        features, targets, token = self._load_scene_with_token(self.tokens[idx])

        if "images" in features:
            images = self._maybe_load_images_from_paths(features["images"])
        elif "image_paths" in features:
            images = self._maybe_load_images_from_paths(features["image_paths"])
        else:
            raise KeyError("Cached features must contain either 'images' or 'image_paths'.")

        features["images"] = images
        if "image_paths" in features:
            del features["image_paths"]

        if "future_frames" in targets:
            future = self._maybe_load_images_from_paths(targets["future_frames"])
        elif "future_image_paths" in targets:
            future = self._maybe_load_images_from_paths(targets["future_image_paths"])
        else:
            raise KeyError("Cached features must contain either 'future_image_paths' or 'future_frames'.")

        if future is not None and isinstance(future, torch.Tensor) and future.ndim == 4:
            targets["future_frames"] = future
        if "future_image_paths" in targets:
            del targets["future_image_paths"]

        return features, targets

# =========================================
# Collate: NavSim -> batch for training loop
# =========================================

def _to_tensor(x, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(dtype=dtype)
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=dtype)
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype=dtype)
    raise TypeError(f"Unsupported type to tensor: {type(x)}")


def navsim_genie_collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
):
    """
    Collate function for NavSim data.
    Inputs:
      features[i]["images"]            : (T_hist, C, H, W) in [-1, 1]
      targets[i]["trajectory"]         : (L, C_action)
      targets[i].get("future_frames")  : optional (T_fut, C, H, W) populated by TrajectoryTargetBuilder.
    Outputs:
      video        : (B, C, 1, T_hist, H, W)
      actions      : (B, L, C_action)
      caption      : list[str] of length B
      future_video : (B, C, 1, T_fut, H, W) if every element in the batch has future_frames.
    """
    features_list, targets_list = zip(*batch)

    # History video -> (B, C, 1, T, H, W)
    vids = []
    for f in features_list:
        frames = f["images"]                           # (T,C,H,W)
        frames = _resize_to_hw(frames, height=704, width=1280)
        T, C, H, W = frames.shape
        vids.append(frames.permute(1, 0, 2, 3).unsqueeze(1).unsqueeze(0).contiguous())
    video = torch.cat(vids, dim=0)                     # (B,C,1,T,H,W)

    # Captions
    captions = []
    for f in features_list:
        hist = f['history_trajectory']
        cmd  = f['driving_command']
        vel = f['vel']
        acc_vec = f['acc']
        spd = float(torch.linalg.norm(vel).item())
        acc = float(torch.linalg.norm(acc_vec).item())
        captions.append(build_prompt_fixed(hist, cmd, spd, acc))

    # Actions supervision
    actions = torch.stack([_to_tensor(t['trajectory']) for t in targets_list], dim=0)  # (B,L,Ca)
    actions = norm_odo(actions)

    # Optional future video
    have_all_future = all(('future_frames' in t and t['future_frames'] is not None) for t in targets_list)
    if have_all_future:
        fut_vids = []
        for t in targets_list:
            fut = t['future_frames']                   # (Tf,C,H,W)
            fut = _resize_to_hw(fut, height=704, width=1280)
            Tf, C, H, W = fut.shape
            fut_vids.append(fut.permute(1, 0, 2, 3).unsqueeze(1).unsqueeze(0).contiguous())
        future_video = torch.cat(fut_vids, dim=0)      # (B,C,1,Tf,H,W)

    # Also batch meta information and return it.
    hist_trajs = torch.stack([_to_tensor(f['history_trajectory']) for f in features_list], dim=0)  # (B,Th,3)
    cmds = torch.stack([_to_tensor(f['driving_command']) for f in features_list], dim=0)           # (B,C_cmd)
    vels = torch.stack([_to_tensor(f['vel']) for f in features_list], dim=0)   # (B,2)
    accs = torch.stack([_to_tensor(f['acc']) for f in features_list], dim=0)   # (B,2)

    out = {
        "video": video,                 # (B,C,1,T,H,W)
        "caption": captions,            # list[str]
        "actions": actions,             # (B,L,Ca)  normalized
        "history_trajectory": hist_trajs,  # (B,Th,3)
        "driving_command": cmds,        # (B,C_cmd)
        "vel": vels,                    # (B,2)
        "acc": accs,                    # (B,2)
    }
    out["future_video"] = future_video

    return out

def norm_odo(trajectory: torch.Tensor) -> torch.Tensor:
        """Normalizes trajectory coordinates and heading to the range [-1, 1]."""
        x = 2 * (trajectory[..., 0:1] + 1.57) / 66.74 - 1
        y = 2 * (trajectory[..., 1:2] + 19.68) / 42 - 1
        heading = 2 * (trajectory[..., 2:3] + 1.67) / 3.53 - 1
        return torch.cat([x, y, heading], dim=-1)

def denorm_odo(normalized_trajectory: torch.Tensor) -> torch.Tensor:
    """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
    x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
    y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
    heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
    return torch.cat([x, y, heading], dim=-1)

def _pad_to_8n1(x_5d: torch.Tensor) -> torch.Tensor:
    """
    Pads a temporal sequence so that the final length satisfies 8n+1 by
    repeating the last frame at the tail.
    Input:  x_5d with shape (B*, C, T, H, W).
    Output: tensor with shape (B*, C, T', H, W), where T' satisfies 8n+1.
    """
    T = x_5d.shape[2]
    need = (1 - (T % 8)) % 8  # ensures (T + need) % 8 == 1
    if need > 0:
        tail = x_5d[:, :, -1:, :, :].repeat(1, 1, need, 1, 1)
        x_5d = torch.cat([x_5d, tail], dim=2)
    return x_5d


def _to_8n1(video_5d: torch.Tensor, ctx_5d: torch.Tensor = None):
    """
    Ensure a temporal length that satisfies 8n+1, optionally with a context frame.
    Args:
        video_5d: (B*V, C, T, H, W) future sequence.
        ctx_5d  : (B*V, C, 1, H, W) context (last history frame), or None.

    Returns:
        (B*V, C, T', H, W) where T' satisfies 8n+1.
    """
    assert video_5d.dim() == 5
    if ctx_5d is not None:
        assert ctx_5d.shape[:2] == video_5d.shape[:2] and ctx_5d.shape[3:] == video_5d.shape[3:]
        video_5d = torch.cat([ctx_5d, video_5d], dim=2)

    T = video_5d.shape[2]
    r = (1 - (T % 8)) % 8            # number of frames to pad so that (T + r) % 8 == 1
    if r > 0:
        tail = video_5d[:, :, -1:, :, :].repeat(1, 1, r, 1, 1)
        video_5d = torch.cat([video_5d, tail], dim=2)
    return video_5d

# ===========================
# Unified trainer (rewritten)
# ===========================
class State:
    seed: int = None
    accelerator: Accelerator = None
    weight_dtype: torch.dtype = None
    train_epochs: int = None
    train_steps: int = None
    num_updates_per_epochs: int = None
    overwrote_max_train_steps: bool = False
    num_trainable_parameters: int = 0
    learning_rate: float = None
    train_batch_size: int = None
    generator: torch.Generator = None
    output_dir: str = None


class UnifiedTrainer:
    """
    Unified trainer.
      - Data: from NavSim (SceneLoader/Dataset are built in prepare_dataset/prepare_val_dataset).
      - Model/training loop: reuses diffusion training logic (video/action branches, flow matching, accelerate).
      - Text: captions are generated by the collate function using a fixed template.
    """

    def __init__(self, cfg,config_file):
        """
        cfg: Hydra training configuration, expected to contain:
          - NavSim paths and splits:
              navsim_log_path, sensor_blobs_path, train_logs, val_logs,
              train_test_split.scene_filter, cache_path, force_cache_computation
          - dataloader.params: {batch_size, num_workers, ...}
          - genie_config: model/optimization yaml path.
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

        self.args.enable_val = getattr(self.args, "enable_val", False)
        # Logging and checkpoint directories.
        if self.state.accelerator.is_main_process:
            start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.save_folder = os.path.join(self.cfg.output_dir, start_time)
            os.makedirs(self.save_folder, exist_ok=True)
        else:
            self.save_folder = self.cfg.output_dir

        # ========== EMA configuration ==========
        # Keep default behavior / hyperparameter semantics aligned with run_training_videodrive_ema.py.
        self.use_ema = bool(getattr(self.args, "use_ema", True))
        self.ema_decay = float(getattr(self.args, "ema_decay", 0.9999))
        self.ema_update_after_step = int(getattr(self.args, "ema_update_after_step", 0))  # 0 => update from the beginning.
        self.ema_inv_gamma = float(getattr(self.args, "ema_inv_gamma", 1.0))              # warmup schedule.
        self.ema_power = float(getattr(self.args, "ema_power", 2 / 3))                    # warmup schedule.
        self.ema_min_decay = float(getattr(self.args, "ema_min_decay", 0.0))

        print(
            f"[EMA Config] use_ema={self.use_ema}, "
            f"ema_decay={self.ema_decay}, ema_min_decay={self.ema_min_decay}, "
            f"ema_update_after_step={self.ema_update_after_step}, "
            f"use_ema_warmup={getattr(self.args, 'use_ema_warmup', False)}, "
            f"ema_inv_gamma={self.ema_inv_gamma}, ema_power={self.ema_power}"
        )

    # -------- Initialization ----------
    def _init_distributed(self):
        logging_dir = Path(self.cfg.output_dir, "./logging_dir")
        project_config = ProjectConfiguration(project_dir=self.cfg.output_dir, logging_dir=logging_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_pg_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout))
        mixed_precision = "no" if torch.backends.mps.is_available() else 'bf16'
        report_to = None if str(getattr(self.args, "report_to", "none")).lower() == "none" else self.args.report_to

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

    # -------- Dataset construction ----------
    def prepare_dataset(self):
        logger.info("Building NavSim training dataset")
        #agent: AbstractAgent = instantiate(self.cfg.agent)
        use_cache_without_dataset = True
        if use_cache_without_dataset:
            self.train_dataset = CacheOnlyDatasetLazyIO(
                cache_path=self.cfg.cache_path,
                feature_builders=self.agent.get_feature_builders(),
                target_builders=self.agent.get_target_builders(),
                log_names=self.cfg.train_logs,
                min_hist_frames=8,
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
                #cache_path=self.cfg.cache_path,
                force_cache_computation=self.cfg.force_cache_computation,
                is_decoder=True
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
        use_cache_without_dataset = True
        if use_cache_without_dataset:
            self.val_dataset = CacheOnlyDatasetLazyIO(
                cache_path=self.cfg.cache_path,
                feature_builders=self.agent.get_feature_builders(),
                target_builders=self.agent.get_target_builders(),
                log_names=self.cfg.val_logs,
                min_hist_frames=8,
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
                #cache_path=self.cfg.cache_path,
                force_cache_computation=self.cfg.force_cache_computation,
                is_decoder=True
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

    # -------- Models / optimizer, etc. ----------
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
        
        text_uncond = get_text_conditions(self.tokenizer, self.text_encoder, prompt="worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, temporal inconsistency, jerky motion, choppy framerate")
        self.uncond_prompt_embeds = text_uncond['prompt_embeds']
        self.uncond_prompt_attention_mask = text_uncond['prompt_attention_mask']

        self.SPATIAL_DOWN_RATIO  = getattr(self.agent, "SPATIAL_DOWN_RATIO", self.vae.spatial_compression_ratio)
        self.TEMPORAL_DOWN_RATIO = getattr(self.agent, "TEMPORAL_DOWN_RATIO", self.vae.temporal_compression_ratio)

        sampler_cls = SAMPLERS["shifted_logit_normal"]
        self._timestep_sampler = sampler_cls()


    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")
        if self.args.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        # if self.args.gradient_checkpointing:
        #     self.diffusion_model.enable_gradient_checkpointing()

        if self.args.allow_tf32 and torch.cuda.is_available():
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

        if self.state.accelerator.is_main_process:
            def _format_param(name: str, p: torch.nn.Parameter) -> str:
                try:
                    shape_str = str(tuple(p.shape))  # e.g. "(512, 2048)"
                except Exception:
                    shape_str = "<no-shape>"
                try:
                    n = p.numel()
                except Exception:
                    n = 0
                return f"{name:<80} | shape={shape_str:<22} | numel={n:>12d} | requires_grad={p.requires_grad}"

            trainables = [(n, p) for n, p in self.diffusion_model.named_parameters() if p.requires_grad]
            frozens    = [(n, p) for n, p in self.diffusion_model.named_parameters() if not p.requires_grad]

            os.makedirs(self.save_folder, exist_ok=True)
            out_txt = os.path.join(self.save_folder, "trainable_params.txt")
            with open(out_txt, "w") as f:
                f.write("=== Trainable ===\n")
                for n, p in trainables:
                    f.write(_format_param(n, p) + "\n")
                f.write("\n=== Frozen ===\n")
                for n, p in frozens:
                    f.write(_format_param(n, p) + "\n")
            logger.info(f"Wrote detailed parameter list to {out_txt}")

        self.state.learning_rate = 3e-5
        # if self.cfg.scale_lr:
        #     self.state.learning_rate = (
        #         self.state.learning_rate
        #         * self.cfg.gradient_accumulation_steps
        #         * self.cfg.dataloader.params.batch_size
        #         * self.state.accelerator.num_processes
        #     )

        params_to_optimize = [{"params": trainable_params, "lr": self.state.learning_rate}]
        self.optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=3e-5,
            beta1=0.9,
            beta2=0.95,
            beta3=0.999,
            epsilon=1e-8,
            weight_decay=1e-5,
            use_8bit=getattr(self.args, "optimizer_8bit", False),
            use_torchao=getattr(self.args, "optimizer_torchao", False),
        )

        # steps
        dataset_size  = len(self.train_dataset)  
        per_device_bs = self.cfg.dataloader.params.batch_size 
        world_size    = self.state.accelerator.num_processes 
        grad_accum    = self.cfg.gradient_accumulation_steps 

        num_upd_per_epoch = math.ceil(dataset_size / (per_device_bs * world_size * grad_accum))
        self.state.num_updates_per_epoch = num_upd_per_epoch

        if self.state.train_steps is None:
            self.state.train_steps = self.args.train_steps or (self.args.train_epochs * num_upd_per_epoch)
            self.state.overwrote_max_train_steps = True
        self.state.train_epochs = self.args.train_epochs

        num_training_steps = int(self.state.train_steps)           
        num_warmup_steps   = int(self.args.lr_warmup_steps)    

        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=1,
            power=1.0,
        )
        #self.state.num_updates_per_epoch = num_upd_per_epoch

        # ========== EMA model initialization (aligned with run_training_videodrive_ema.py) ==========
        self.ema = None
        self._ema_params = None
        if self.use_ema:
            base_model = unwrap_model(self.state.accelerator, self.diffusion_model)
            # Track parameters in a consistent order (only those with requires_grad=True).
            self._ema_params = [p for p in base_model.parameters() if p.requires_grad]
            self.ema = EMAModel(
                parameters=self._ema_params,
                decay=self.ema_decay,
                min_decay=self.ema_min_decay,
                update_after_step=self.ema_update_after_step,
                use_ema_warmup=True,
                inv_gamma=self.ema_inv_gamma,
                power=self.ema_power,
            )
            # Move EMA to the correct device (safe with ZeRO/offload).
            try:
                self.ema.to(self.state.accelerator.device)
            except Exception:
                pass

    def prepare_for_training(self):
        self.diffusion_model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = \
            self.state.accelerator.prepare(self.diffusion_model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler)

    def prepare_trackers(self):
        self.state.accelerator.init_trackers("navsim_train", config=self.args.__dict__)

    def _create_first_frame_conditioning_mask(
        self, batch_size: int, sequence_length: int, height: int, width: int, device: torch.device
    ) -> Tensor:
        """Create conditioning mask for first frame conditioning.

        Returns:
            Boolean mask where True indicates first frame tokens (if conditioning is enabled)
        """
        conditioning_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=device)

        first_frame_conditioning_p = 1
        if (
            first_frame_conditioning_p > 0
            and random.random() < first_frame_conditioning_p
        ):
            first_frame_end_idx = height * width
            if first_frame_end_idx < sequence_length:
                conditioning_mask[:, :first_frame_end_idx] = True

        return conditioning_mask

    def _create_timesteps_from_conditioning_mask(
        self, conditioning_mask: Tensor, sampled_timestep_values: Tensor
    ) -> Tensor:
        """Create timesteps based on conditioning mask.

        Args:
            conditioning_mask: Boolean mask of shape (batch_size, sequence_length),
            where True = conditioning, False = target.
            sampled_timestep_values: Sampled timestep values for target tokens of shape (batch_size,)

        Returns:
            Timesteps tensor with 0 for conditioning tokens, sampled values for target tokens
        """
        # Expand sampled values to match conditioning mask shape
        expanded_timesteps = sampled_timestep_values.unsqueeze(1).expand_as(conditioning_mask)

        # Use conditioning mask to select between 0 (conditioning) and sampled values (target)
        return torch.where(conditioning_mask, 0, expanded_timesteps)

    # -------- Training ----------
    def train(self):
        logger.info("Starting training")
        logger.info("Memory before: %s", json.dumps(get_memory_statistics(), indent=2))

        accel = self.state.accelerator

        per_device_bs = self.cfg.dataloader.params.batch_size
        world_size    = accel.num_processes
        grad_accum    = self.cfg.gradient_accumulation_steps
        global_bs     = per_device_bs * world_size * grad_accum

        logger.info(f"Effective Global Batch Size = {global_bs} "
                    f"(= {per_device_bs} per_device x {world_size} world x {grad_accum} grad_accum)")
        logger.info(f"Num updates / epoch       = {self.state.num_updates_per_epoch}")
        logger.info(f"Train epochs              = {self.state.train_epochs}")
        logger.info(f"Total train steps         = {self.state.train_steps}")

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
                with accel.accumulate(self.diffusion_model):
                    # 1) Use history frames and future frames (if available).
                    video = batch['video'].to(accel.device, dtype=weight_dtype).contiguous()   # (B, C, 1, T_hist, H, W)
                    b, c, v, t, h, w = video.shape
                    assert t == 4
                    video = rearrange(video, 'b c v t h w -> (b v) c t h w')                   # (B*V, C, T_hist, H, W)
                    mem = video

                    future_video = batch['future_video'].to(accel.device, dtype=weight_dtype).contiguous()  # (B, C, 1, T_fut, H, W)
                    future_video = rearrange(future_video, 'b c v t h w -> (b v) c t h w')
                    assert future_video.shape[2] == 8, f"Expected 8 future frames, got {future_video.shape[2]}"
                    # Take the last frame and repeat once to get 9 frames.
                    last_frame = future_video[:, :, -1:, :, :]  # (B*V, C, 1, H, W)
                    future_video = torch.cat([future_video, last_frame], dim=2)  # (B*V, C, 9, H, W)
                    fut = future_video

                    if getattr(self.args, "use_color_jitter", False):
                        fut = apply_color_jitter_to_video(fut)

                    mem_latents, future_video_latents = get_latents(
                        self.vae, mem, fut
                    )

                    latent_frames = 4 + 9 // self.TEMPORAL_DOWN_RATIO + 1
                    latent_height = h // self.SPATIAL_DOWN_RATIO
                    latent_width = w // self.SPATIAL_DOWN_RATIO

                    mem_latents = rearrange(mem_latents, '(b m) (h w) c -> b c m h w', b=b, m=4, h=latent_height)
                    #mem_latents = rearrange(mem_latents, 'b (f h w) c -> b c f h w', b=b,h=latent_height,w=latent_width)
                    future_video_latents = rearrange(future_video_latents, 'b (f h w) c -> b c f h w',b=b,h=latent_height,w=latent_width)
                    latents = torch.cat((mem_latents, future_video_latents), dim=2)

                    video_attention_mask = None
                    latents = rearrange(latents, 'b c f h w -> b (f h w) c')


                    cmd_onehot = batch['driving_command'].to(accel.device, dtype=weight_dtype)      # (B,Ccmd)
                    vel = batch['vel'].to(accel.device, dtype=weight_dtype)                    # (B,2)
                    acc_vec = batch['acc'].to(accel.device, dtype=weight_dtype)                # (B,2)

                    context_dict = {
                        "cmd_onehot": cmd_onehot,
                        "vel": vel,
                        "acc": acc_vec,
                    }
                    # 2) Text conditioning with dropout.
                    captions = batch['caption']
                    dropout_factor = torch.rand(b, device=accel.device, dtype=weight_dtype)
                    dropout_mask_prompt = (dropout_factor < self.args.caption_dropout_p).unsqueeze(1).unsqueeze(2)
                    text_conds = get_text_conditions(self.tokenizer, self.text_encoder, captions)
                    prompt_embeds = text_conds['prompt_embeds']
                    prompt_attention_mask = text_conds['prompt_attention_mask']
                    prompt_embeds = self.uncond_prompt_embeds.repeat(b, 1, 1) * dropout_mask_prompt + \
                                    prompt_embeds * ~dropout_mask_prompt


                    # 4) Flow-matching on the video side (used for noise/timestep sampling).
                    # action_weights = compute_density_for_timestep_sampling(
                    #     weighting_scheme=self.args.flow_weighting_scheme,
                    #     batch_size=b,
                    #     logit_mean=self.args.flow_logit_mean,
                    #     logit_std=self.args.flow_logit_std,
                    #     mode_scale=self.args.flow_mode_scale,
                    # )
                    # action_indices   = (action_weights * self.scheduler.config.num_train_timesteps).long()
                    # action_sigmas    = scheduler_sigmas[action_indices]
                    # action_timesteps = (action_sigmas * 1000.0).long()# .unsqueeze(-1).repeat(1, actions.shape[1]

                    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                        mem_latents, latent_frames, latent_height, latent_width,
                        noise_to_condition_frames=0.0
                    )  # set initial frames noise to 0

                    noisy_latents = noise
                    sigmas_vec = self._timestep_sampler.sample_for(noisy_latents)
                    sigmas = sigmas_vec                                            # (b,)
                    timesteps = torch.round(sigmas_vec * 1000.0).long()            # (b,)

                    if self.args.pixel_wise_timestep:
                        # shape: b, thw
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask.long())
                    else:
                        # shape: b, t
                        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

                    # shape: b,1,c
                    ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    if self.args.return_action and self.args.noisy_video:
                        ss = torch.full_like(ss, 1.0)

                    noisy_latents = (1.0 - ss) * latents + ss * noise



                    actions = batch['actions'].to(accel.device, dtype=weight_dtype)            # (B,L,Ca)
                    action_dim = actions.shape[-1]
                    noise_actions = randn_tensor(actions.shape, device=accel.device, dtype=weight_dtype)
                    
                    t_actions = self._timestep_sampler.sample_for(actions)            # (B,)
                    t_actions = t_actions.clamp(0.0, 1.0)

                    t_b11        = t_actions.view(-1, 1, 1)                           # (B,1,1)
                    noisy_actions = (1.0 - t_b11) * actions + t_b11 * noise_actions   # (B, L, Ca)

                    NUM_T_STEPS = getattr(self.scheduler.config, "num_train_timesteps", 1000)
                    action_timesteps = torch.round(t_actions * 1000).long()       # (B,)
                    action_timesteps = action_timesteps.unsqueeze(-1).expand(-1, actions.shape[1])  # (B, L)

                    action_loss_weight = 1.0


                    pred_all = self._forward_pass(
                        latents=noisy_latents.to(accel.device, dtype=weight_dtype),
                        timesteps=timesteps,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        latent_frames=latent_frames,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        return_video=self.args.return_video or self.args.return_action,
                        return_action=self.args.return_action,
                        video_attention_mask=None,
                        action_timestep=action_timesteps,
                        action_states=noisy_actions.to(accel.device, dtype=weight_dtype),
                        action_dim=action_dim,
                        context_dict=context_dict,
                    )
                    pred_vel = pred_all['action']                                              # (B, L, Ca)

                    # 7) Action loss.
                    target_vel = noise_actions - actions
                    loss_action = ((pred_vel - target_vel).pow(2)).mean()
                    loss = loss_action
                    assert not torch.isnan(loss), "NaN loss detected"
                    accel.backward(loss)
                    if accel.sync_gradients and accel.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accel.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                        logs["grad_norm"] = grad_norm
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    # EMA update (only when gradients are synchronized).
                    if self.use_ema and accel.sync_gradients:
                        base = unwrap_model(accel, self.diffusion_model)
                        ema_params_now = [p for p in base.parameters() if p.requires_grad]
                        self.ema.step(ema_params_now)

                    self.optimizer.zero_grad()
                # Logging
                loss = accel.reduce(loss.detach(), reduction='mean')
                if self.args.train_mode in ('all','action_only','action_full'):
                    loss_action = accel.reduce(loss_action.detach(), reduction='mean')
                # if self.args.train_mode in ('all','video_only'):
                #     loss_video = accel.reduce(loss_video.detach(), reduction='mean')

                running_loss += loss.item()
                if accel.sync_gradients:
                    global_step += 1

                logs.update({"loss": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]})
                if (global_step % self.args.steps_to_log == 0) or (step == 0):
                    accel.log(logs, step=global_step)
                    if accel.is_main_process:
                        logger.info(f"[step {global_step}] loss={logs['loss']:.4f}, lr={logs['lr']:.6g}")

                # Validation
                if (self.args.enable_val 
                    and self.val_dataloader is not None 
                    and global_step % self.args.steps_to_val == 0):
                    accel.wait_for_everyone()
                    # Use EMA weights for validation (consistent with run_training_videodrive_ema.py).
                    if self.use_ema:
                        model_real = unwrap_model(accel, self.diffusion_model)
                        ema_params_now = [p for p in model_real.parameters() if p.requires_grad]
                        self.ema.store(ema_params_now)
                        self.ema.copy_to(ema_params_now)

                    model_save_dir = os.path.join(self.save_folder, f'Validation_step_{global_step}')
                    self.validate(accel, model_save_dir, global_step)

                    if self.use_ema:
                        self.ema.restore(ema_params_now)

                    accel.wait_for_everyone() 
                # Saving
                if global_step % self.args.steps_to_save == 0 and accel.is_main_process:
                    model_to_save = unwrap_model(accel, self.diffusion_model)
                    model_save_dir = os.path.join(self.save_folder, f'step_{global_step}')
                    model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                    # Additionally save EMA weights.
                    if self.use_ema:
                        ema_params_now = [p for p in model_to_save.parameters() if p.requires_grad]
                        self.ema.store(ema_params_now)
                        self.ema.copy_to(ema_params_now)
                        model_save_dir_ema = os.path.join(self.save_folder, f'step_{global_step}_ema')
                        model_to_save.save_pretrained(model_save_dir_ema, safe_serialization=True)
                        self.ema.restore(ema_params_now)
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
            # 最终再导出一份 EMA 权重
            if self.use_ema:
                ema_params_now = [p for p in self.diffusion_model.parameters() if p.requires_grad]
                self.ema.store(ema_params_now)
                self.ema.copy_to(ema_params_now)
                final_dir_ema = os.path.join(self.save_folder, f'step_{global_step}_ema')
                self.diffusion_model.save_pretrained(final_dir_ema, safe_serialization=True)
                self.ema.restore(ema_params_now)

        del self.diffusion_model, self.scheduler
        free_memory()
        logger.info(f"Memory after training: {json.dumps(get_memory_statistics(), indent=2)}")
        accel.end_training()

    def _forward_pass(self, latents, timesteps, prompt_embeds, prompt_attention_mask,
                      latent_frames, latent_height, latent_width,
                      return_video, return_action, video_attention_mask,
                      action_timestep=None, action_states=None, action_dim=None,
                      context_dict = None,rope_interpolation_scale=None):
        """Thin wrapper compatible with utils.forward_pass interface and return format."""
        return forward_pass(
            model=self.diffusion_model,
            timesteps=timesteps,
            noisy_latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            num_frames=latent_frames,
            height=latent_height,
            width=latent_width,
            action_states=action_states,
            action_timestep=action_timestep,
            return_video=return_video,
            return_action=return_action,
            video_attention_mask=video_attention_mask,
            context_dict=context_dict,
            rope_interpolation_scale=rope_interpolation_scale,
        )['latents']

    @staticmethod
    def _tensor_video_to_pils(video: torch.Tensor):
        """
        video: (T, C, H, W) in [-1, 1] or [0, 1]
        return: List[PIL.Image]
        """
        x = video.detach().cpu()
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        pils = []
        T = x.shape[0]
        for t in range(T):
            arr = (x[t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            pils.append(Image.fromarray(arr))
        return pils
    
    def _pad_to_8n1(self, x_5d: torch.Tensor) -> torch.Tensor:
        """
        x_5d: (B*, C, T, H, W) temporal sequence.
        Pads the tail by repeating the last frame so that T' satisfies 8n+1.
        """
        T = x_5d.shape[2]
        need = (1 - (T % 8)) % 8  # ensures (T + need) % 8 == 1
        if need > 0:
            tail = x_5d[:, :, -1:, :, :].repeat(1, 1, need, 1, 1)
            x_5d = torch.cat([x_5d, tail], dim=2)
        return x_5d

    # -------- Validation ----------
    @torch.inference_mode()
    def validate(self, accelerator, model_save_dir, global_step):
        """
        Validate actions using the current NavSim collate format:
        - Inputs: full history from batch['video'].
        - Targets: batch['actions'].
        - Metric: MSE (L2) over the full validation set.
        - Artifacts: save predictions and ground truth to model_save_dir.
        """
        os.makedirs(model_save_dir, exist_ok=True)

        # Pipeline (unwrap deepspeed wrapper if used).
        pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer,
            unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model
        )

        # Basic inference hyperparameters (respect existing config fields).
        num_steps   = 5
        guidance    = float(getattr(self.args, "guidance_scale", 1.0))
        pixel_wise  = bool(getattr(self.args, "pixel_wise_timestep", True))
        seed        = int(getattr(self.args, "seed", 42))
        negative_prompt = getattr(self.args, "negative_prompt",
            "worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, "
            "flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, "
            "temporal inconsistency, jerky motion, choppy framerate"
        )
        device = self.state.accelerator.device
        generator = torch.Generator(device=device).manual_seed(seed)

        weight_dtype = self.state.weight_dtype

        total_mse = 0.0
        total_count = 0
        total_l1  = 0.0 


        pbar = tqdm(
            total=len(self.val_dataloader),
            desc=f"Validating @ step {global_step}",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True
        )

        for batch in self.val_dataloader:
            # History condition: (B, C, 1, T, H, W) -> (B, C, T, H, W)
            video = batch['video']  # shape (B,C,1,T_hist,H,W)
            B, C, V, T_hist, H, W = video.shape
            assert T_hist == 4
            assert V == 1, f"Validation currently assumes a single view, got V={V}"

            prompts = batch['caption']            # list[str], len=B
            gt_actions = batch['actions']         # (B,L,Ca)
            action_chunk = gt_actions.shape[1]
            action_dim   = gt_actions.shape[2]

            hist_traj = batch['history_trajectory']        # (B,Th,3)
            cmd_onehot = batch['driving_command']      # (B,Ccmd)
            vel = batch['vel']                    # (B,2)
            acc_vec = batch['acc']                # (B,2)

            for i in range(B):
                cond_tensor = video[i].squeeze(1)       # (C, T, H, W)
                cond_tensor = rearrange(cond_tensor, 'c t h w -> t c h w').contiguous()  # (T, C, H, W)

                cond_pils = self._tensor_video_to_pils(cond_tensor)  
                condition = LTXVideoCondition(video=cond_pils, frame_index=0)

                prompt_i = [prompts[i]]


                cmd_onehot_b1 = batch['driving_command'][i:i+1].to(device, dtype=weight_dtype)        # (1, C_cmd)
                vel_b1        = batch['vel'][i:i+1].to(device, dtype=weight_dtype)                    # (1, 2)
                acc_b1        = batch['acc'][i:i+1].to(device, dtype=weight_dtype)                    # (1, 2)

                context_dict = {
                    "cmd_onehot": cmd_onehot_b1,
                    "vel":        vel_b1,
                    "acc":        acc_b1,
                }

                out = pipe.infer(
                    conditions=[condition],
                    prompt=prompt_i,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_steps,
                    guidance_scale=1.0,
                    height=704,
                    width=1280,
                    generator=generator,
                    num_frames=18,
                    output_type = "latent",
                    return_action=True,
                    return_video=False,          # only actions are needed
                    action_chunk=action_chunk,   # aligned with GT horizon
                    action_dim=action_dim,
                    context_dict=context_dict,
                    pixel_wise_timestep = self.args.pixel_wise_timestep,
                    noise_seed=42,
                    image_cond_noise_scale=0.0
                )

                if isinstance(out, (list, tuple)):
                    out = out[0]
                if isinstance(out, dict):
                    pred_action = out['actions']
                else:
                    pred_action = out
                pred_action = torch.as_tensor(pred_action).detach().float().cpu()
                if pred_action.ndim == 3:  # (B,T,C) -> (T,C)
                    pred_action = pred_action[0]

                pred_action = pred_action
                pred_action = denorm_odo(pred_action)
                gt_i = denorm_odo(gt_actions[i].detach().float().cpu())

                mse_i  = ((pred_action - gt_i)**2).mean().item()
                l1_i  = (pred_action - gt_i).abs().mean().item()  

                total_mse += mse_i
                total_l1  += l1_i          
                total_count += 1
            
            if accelerator.is_main_process:
                cur_mean_mse = total_mse / max(total_count, 1)
                cur_mean_l1  = total_l1  / max(total_count, 1)    # <<< NEW
                pbar.set_postfix_str(f"mean_l2={cur_mean_mse:.4f}, mean_l1={cur_mean_l1:.4f}")  # <<< NEW
                pbar.update(1)

        pbar.close()

        # Reduce triple: mse, l1, count.
        local_sum = torch.tensor([total_mse, total_l1, float(total_count)], device=accelerator.device, dtype=torch.float32)  # <<< NEW
        global_sum = accelerator.reduce(local_sum, reduction="sum")
        denom = torch.clamp(global_sum[2], min=1.0)  # count
        mean_mse = (global_sum[0] / denom).item()
        mean_l1  = (global_sum[1] / denom).item()    # <<< NEW

        if accelerator.is_main_process:
            accelerator.log({"val/action_l2": mean_mse, "val/action_l1": mean_l1}, step=global_step)  # <<< NEW
            print("val/action_l2", mean_mse)
            print("val/action_l1", mean_l1)  # <<< NEW

        return {
            "val_action_l2": mean_mse,
            "val_action_l1": mean_l1,  # <<< NEW
        }


def pack_latents(
    latents: Tensor,
    spatial_patch_size: int = 1,
    temporal_patch_size: int = 1,
) -> Tensor:
    """Reshapes latents [B,C,F,H,W] into patches and flattens to sequence form [B,L,D].

    Args:
        latents: Input latent tensor
        spatial_patch_size: Size of spatial patches
        temporal_patch_size: Size of temporal patches

    Returns:
        Flattened sequence of patches
    """
    b, c, f, h, w = latents.shape
    latents = latents.reshape(
        b,
        -1,
        f // temporal_patch_size,
        temporal_patch_size,
        h // spatial_patch_size,
        spatial_patch_size,
        w // spatial_patch_size,
        spatial_patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def encode_video(
    vae: AutoencoderKLLTXVideo,
    image_or_video: Tensor,
    patch_size: int = 1,
    patch_size_t: int = 1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> dict[str, Tensor | int]:
    """Encodes input images/videos into latent representations.

    Args:
        vae: VAE model for encoding
        image_or_video: Input tensor of shape [B,C,F,H,W] or [B,C,1,H,W]
        patch_size: Spatial patch size
        patch_size_t: Temporal patch size
        device: Target device for tensors
        dtype: Target dtype for tensors
        generator: Random number generator

    Returns:
        Dict containing latents and shape information
    """
    device = device or vae.device

    if image_or_video.ndim == 4:
        image_or_video = image_or_video.unsqueeze(2)
    assert image_or_video.ndim == 5, f"Expected 5D tensor, got {image_or_video.ndim}D tensor"

    image_or_video = image_or_video.to(device=device, dtype=vae.dtype)
    #image_or_video = image_or_video.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, F, H, W] -> [B, F, C, H, W]
    image_or_video = image_or_video.contiguous()
    # Encode image/video.
    latents = vae.encode(image_or_video).latent_dist.sample(generator=generator)
    latents = latents.to(dtype=dtype)
    _, _, num_frames, height, width = latents.shape

    # Normalize to zero mean and unit variance.
    latents = _normalize_latents(latents, vae.latents_mean, vae.latents_std)

    # Patchify and pack latents to a sequence expected by the transformer.
    latents = pack_latents(latents, patch_size, patch_size_t)
    return {"latents": latents, "num_frames": num_frames, "height": height, "width": width}


def _normalize_latents(
    latents: Tensor,
    mean: Tensor,
    std: Tensor,
) -> Tensor:
    """Normalizes latents using mean and standard deviation across the channel dimension."""
    mean = mean.view(1, -1, 1, 1, 1).repeat(latents.shape[0], 1, 1, 1, 1).to(latents.device, latents.dtype)
    std = std.view(1, -1, 1, 1, 1).repeat(latents.shape[0], 1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents - mean) / std
    return latents

def get_rope_scale_factors(fps: float) -> list[float]:
    """Get ROPE interpolation scale factors for video transformers.

    Args:
        fps: Frames per second

    Returns:
        List of scale factors [temporal_scale, spatial_scale, spatial_scale]
    """
    if fps <= 0:
        raise ValueError("FPS must be a positive number.")

    temporal_compression_ratio = 8.0
    spatial_compression_ratio = 32.0

    return [
        temporal_compression_ratio / fps,
        spatial_compression_ratio,
        spatial_compression_ratio,
    ]

# ===========================
# Hydra main
# ===========================
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    trainer = UnifiedTrainer(cfg,'navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim_stage1.yaml')
    trainer.prepare_dataset()
    trainer.prepare_val_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()  
    trainer.prepare_for_training()         
    trainer.prepare_trackers()
    trainer.train()

if __name__ == "__main__":
    main()
