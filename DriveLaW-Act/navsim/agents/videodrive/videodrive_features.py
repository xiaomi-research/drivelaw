from typing import Dict, Tuple, List, Any, Union,Sequence
import torch
import numpy as np
import gzip
import pickle
from PIL import Image
import cv2
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


class VideoDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        prompt_frames: int = 9,
        normalize: str = "[-1,1]",
        ltx_min_prompt_frames: int = 8,
        view_mode: str = "front",   # "front" or "surround6"
        surround_keys: Sequence[str] = (
            "cam_l0", "cam_f0", "cam_r0",
            "cam_l2", "cam_b0", "cam_r2",
        ),
        surround_grid: Tuple[int, int] = (2, 3),
    ):
        self.prompt_frames = prompt_frames
        assert normalize in ["[0,1]", "[-1,1]"]
        self.normalize = normalize
        self.ltx_min_prompt_frames = int(ltx_min_prompt_frames)
        self.view_mode = view_mode
        self.surround_keys = tuple(surround_keys)
        self.surround_grid = surround_grid

    def get_unique_name(self) -> str:
        return "videodrive_feature"

    def _grab_cam(self, cams):
        f0 = cams.cam_f0
        img = f0.image
        if isinstance(img, (str, Path)):
            return "path", str(img)
        if img is None:
            raise ValueError("cams.cam_f0.image is None")
        color_space = getattr(f0, "color_space", "RGB")
        if color_space.upper() == "BGR" or (img[..., 0].mean() > img[..., 2].mean()):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return "rgb", img

    def _grab_surround(self, cams):
        # Returns ("path", [6 paths]) or ("rgb", mosaic RGB)
        items = []
        modes = []
        for key in self.surround_keys:
            cam = getattr(cams, key)
            img = cam.image
            if isinstance(img, (str, Path)):
                modes.append("path"); items.append(str(img))
            else:
                modes.append("rgb")
                if img is None:
                    raise ValueError(f"{key}.image is None")
                color_space = getattr(cam, "color_space", "RGB")
                if color_space.upper() == "BGR" or (img[..., 0].mean() > img[..., 2].mean()):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                items.append(img)
        # If all are paths, return path list directly; otherwise create mosaic
        if all(m == "path" for m in modes):
            return "path", items
        else:
            # Create 2x3 mosaic from non-path ndarrays
            imgs = [x if isinstance(x, np.ndarray) else cv2.imread(x) for x in items]
            # Align sizes (use first image as reference)
            h0, w0 = imgs[0].shape[:2]
            imgs = [cv2.resize(im, (w0, h0), interpolation=cv2.INTER_AREA) for im in imgs]
            row1 = np.concatenate(imgs[0:3], axis=1)
            row2 = np.concatenate(imgs[3:6], axis=1)
            mosaic = np.concatenate([row1, row2], axis=0)
            return "rgb", mosaic

    def _to_chw_tensor(self, img_rgb_uint8: np.ndarray) -> torch.Tensor:
        img = img_rgb_uint8.astype(np.float32) / 255.0
        if self.normalize == "[-1,1]":
            img = img * 2.0 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        num_hist = min(
            len(agent_input.ego_statuses),
            len(agent_input.cameras),
            agent_input.ego2global_T.shape[0] if getattr(agent_input, "ego2global_T", None) is not None else float("inf"),
        )
        if num_hist <= 0:
            raise ValueError("No history frames available in AgentInput.")

        T_raw = min(self.prompt_frames, num_hist)
        start = num_hist - T_raw
        idx_range = range(start, num_hist)

        # ==== Front view / Surround view modes handled separately ====
        paths_front: List[str] = []
        paths_surround: List[List[str]] = []
        imgs_tensor: List[torch.Tensor] = []
        saw_path = False

        for i in idx_range:
            cams = agent_input.cameras[i]
            if self.view_mode == "front":
                mode, item = self._grab_cam(cams)
                if mode == "path":
                    saw_path = True
                    paths_front.append(item)
                else:
                    imgs_tensor.append(self._to_chw_tensor(item))
            else:  # surround6
                mode, item = self._grab_surround(cams)
                if mode == "path":
                    saw_path = True
                    paths_surround.append(item)  # List[str] length 6
                else:
                    imgs_tensor.append(self._to_chw_tensor(item))

        out: Dict[str, Any] = {}

        if saw_path:
            # If any frame exists as path, return as paths without tensor stacking
            if self.view_mode == "front":
                out["image_paths"] = paths_front                     # List[str]
            else:
                out["image_paths"] = paths_surround         # List[List[str]] 6 views per frame
        else:
            # All in-memory images -> stack tensors and resize
            images = torch.stack(imgs_tensor, dim=0)  # (T,C,H,W)
            images = _resize_to_hw(images, 768, 1344)
            out["images"] = images

        # ===== Rest of context remains unchanged =====
        ego_statuses = agent_input.ego_statuses
        history_trajectory = torch.tensor(
            [[float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])] for e in ego_statuses[:4]],
            dtype=torch.float32
        )
        vel = torch.tensor(agent_input.ego_statuses[-1].ego_velocity, dtype=torch.float32)
        acc = torch.tensor(agent_input.ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        driving_command = torch.tensor(agent_input.ego_statuses[-1].driving_command, dtype=torch.float32)

        out.update({
            "history_trajectory": history_trajectory,
            "vel": vel,
            "acc": acc,
            "driving_command": driving_command,
        })
        return out




class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        normalize: str = "[-1,1]",
        view_mode: str = "front",
        surround_keys: Sequence[str] = (
            "cam_l0", "cam_f0", "cam_r0",
            "cam_l2", "cam_b0", "cam_r2",
        ),
        surround_grid: Tuple[int, int] = (2, 3),
        out_hw: Tuple[int, int] | None = None,
    ):
        self._trajectory_sampling = trajectory_sampling
        self.normalize = normalize
        self.view_mode = view_mode
        self.surround_keys = tuple(surround_keys)
        self.surround_grid = surround_grid
        self.out_hw = out_hw

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def _grab_cam(self, cams):
        f0 = cams.cam_f0
        img = f0.image
        if isinstance(img, (str, Path)):
            return "path", str(img)
        if img is None:
            raise ValueError("cams.cam_f0.image is None")
        color_space = getattr(f0, "color_space", "RGB")
        if color_space.upper() == "BGR" or (img[..., 0].mean() > img[..., 2].mean()):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return "rgb", img

    def _grab_surround(self, cams):
        items = []
        modes = []
        for key in self.surround_keys:
            cam = getattr(cams, key)
            img = cam.image
            if isinstance(img, (str, Path)):
                modes.append("path"); items.append(str(img))
            else:
                modes.append("rgb")
                if img is None:
                    raise ValueError(f"{key}.image is None")
                color_space = getattr(cam, "color_space", "RGB")
                if color_space.upper() == "BGR" or (img[..., 0].mean() > img[..., 2].mean()):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                items.append(img)
        if all(m == "path" for m in modes):
            return "path", items
        else:
            # Mosaic consistent with feature
            imgs = [x if isinstance(x, np.ndarray) else cv2.imread(x) for x in items]
            h0, w0 = imgs[0].shape[:2]
            imgs = [cv2.resize(im, (w0, h0), interpolation=cv2.INTER_AREA) for im in imgs]
            row1 = np.concatenate(imgs[0:3], axis=1)
            row2 = np.concatenate(imgs[3:6], axis=1)
            mosaic = np.concatenate([row1, row2], axis=0)
            return "rgb", mosaic

    def _to_chw_tensor(self, img_rgb_uint8: np.ndarray) -> torch.Tensor:
        if self.out_hw is not None:
            H, W = self.out_hw
            img_rgb_uint8 = cv2.resize(img_rgb_uint8, (W, H), interpolation=cv2.INTER_AREA)
        img = img_rgb_uint8.astype(np.float32) / 255.0
        if self.normalize == "[-1,1]":
            img = img * 2.0 - 1.0
        return torch.from_numpy(img).permute(2, 0, 1).contiguous()

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        fut = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        traj = torch.tensor(fut.poses)

        cams_seq: List[Any] = scene.get_future_frames(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        if len(cams_seq) == 0:
            return {"trajectory": traj}

        got_path = False
        frames_tensor: List[torch.Tensor] = []
        fut_paths_front: List[str] = []
        fut_paths_surround: List[List[str]] = []

        for cams in cams_seq:
            if self.view_mode == "front":
                mode, item = self._grab_cam(cams)
                if mode == "path":
                    got_path = True
                    fut_paths_front.append(item)
                else:
                    frames_tensor.append(self._to_chw_tensor(item))
            else:
                mode, item = self._grab_surround(cams)
                if mode == "path":
                    got_path = True
                    fut_paths_surround.append(item)  # List[str] 长度6
                else:
                    frames_tensor.append(self._to_chw_tensor(item))

        if got_path:
            if self.view_mode == "front":
                return {"trajectory": traj, "future_image_paths": fut_paths_front}
            else:
                return {"trajectory": traj, "future_image_paths": fut_paths_surround}
        else:
            future_frames = torch.stack(frames_tensor, dim=0)
            return {"trajectory": traj, "future_frames": future_frames}


def _resize_to_hw(img: torch.Tensor, height: int = 704, width: int = 1280) -> torch.Tensor:
    """
    Supports:
      - (C,H,W)
      - (T,C,H,W) where T is treated as batch dimension
    Fixed resize to (height, width); returns directly if already at this size.
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