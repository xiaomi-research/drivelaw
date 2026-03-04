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
        view_mode: str = "front",   # "front", "surround6", or "surround8"
        surround_keys: Sequence[str] = (
            "cam_l0", "cam_f0", "cam_r0",
            "cam_l2", "cam_b0", "cam_r2",
        ),  # default 6 views; can be overridden to 8 via parameter
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
        # Return ("path", [paths]) or ("rgb", mosaic RGB image).
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
        # If all are paths, return the list directly; otherwise build a mosaic.
        if all(m == "path" for m in modes):
            return "path", items
        else:
            # Horizontally concatenate non-path arrays into a single-row mosaic (1xN), matching training code.
            imgs = [x if isinstance(x, np.ndarray) else cv2.imread(x) for x in items]
            # Align sizes to the first image, then resize each view to the target size.
            h0, w0 = imgs[0].shape[:2]
            # Target per-view size: 512x256 (W x H).
            target_w, target_h = 512, 256
            imgs = [cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA) for im in imgs]
            # Horizontal concatenation: single row of views.
            pano = np.concatenate(imgs, axis=1)  # (H, N*W, 3)
            return "rgb", pano

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

        # Handle "front" versus surround modes separately.
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
            else:  # surround6 or surround8
                mode, item = self._grab_surround(cams)
                if mode == "path":
                    saw_path = True
                    paths_surround.append(item)  # List[str] of length 6 or 8
                else:
                    imgs_tensor.append(self._to_chw_tensor(item))

        out: Dict[str, Any] = {}

        if saw_path:
            # If any frame is represented by paths, return paths without stacking tensors.
            if self.view_mode == "front":
                out["image_paths"] = paths_front                     # List[str]
            else:
                out["image_paths"] = paths_surround         # List[List[str]] per-frame multi-view paths
        else:
            # All images are in memory: stack into tensor and resize.
            images = torch.stack(imgs_tensor, dim=0)  # (T, C, H, W)
            # Set resize dimensions based on view_mode.
            if self.view_mode == "surround6":
                # Multi-view: horizontally concatenated views, each resized to 512x256.
                images = _resize_to_hw(images, 256, 3072)
            else:
                images = _resize_to_hw(images, 768, 1344)
            out["images"] = images

        # Remaining context kept unchanged.
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
        ),  # default 6 views; can be overridden to 8 via parameter
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
            # Horizontally concatenate (single row of views), consistent with feature builder.
            imgs = [x if isinstance(x, np.ndarray) else cv2.imread(x) for x in items]
            # Target per-view size: 512x256 (W x H).
            target_w, target_h = 512, 256
            imgs = [cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA) for im in imgs]
            # Horizontal concatenation: single row of views.
            pano = np.concatenate(imgs, axis=1)  # (H, N*W, 3)
            return "rgb", pano

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
                    fut_paths_surround.append(item)  # List[str] of length 6 or 8
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


def _resize_to_hw(img: torch.Tensor, height: int = 768, width: int = 1344) -> torch.Tensor:
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