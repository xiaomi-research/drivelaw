import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from PIL import Image

# 导入 diffusers 库
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

# 导入数据集类
try:
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if root_path not in sys.path:
        sys.path.append(root_path)
    from dataset.dataset import TrainDataset
    print("✓ Successfully imported TrainDataset")
except ImportError as e:
    print(f"❌ Error importing custom modules: {e}")
    sys.exit(1)


class LTXInferenceEngine:
    """封装 LTX 推理引擎,使用 diffusers 库"""
    
    def __init__(self, model_path="Lightricks/LTX-Video-0.9.7-dev", 
                 upscaler_path="Lightricks/ltxv-spatial-upscaler-0.9.7",  # ✅ 新增参数
                 device="cuda", 
                 use_upscaler=True):
        """
        初始化 LTX 推理引擎
        
        Args:
            model_path: 模型路径或 HuggingFace 模型 ID
            upscaler_path: 上采样器路径或 HuggingFace 模型 ID  # ✅ 新增
            device: 运行设备
            use_upscaler: 是否使用空间上采样器
        """
        print(f"🔧 Initializing LTX pipeline from {model_path}...")
        
        # 加载主管道
        self.pipe = LTXConditionPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        )
        self.pipe.to(device)
        self.pipe.vae.enable_tiling()
        
        # 可选:加载上采样管道
        self.use_upscaler = use_upscaler
        if use_upscaler:
            print(f"🔧 Loading spatial upscaler from {upscaler_path}...")  # ✅ 显示路径
            self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
                upscaler_path,  # ✅ 使用参数
                vae=self.pipe.vae,
                torch_dtype=torch.bfloat16
            )
            self.pipe_upsample.to(device)
        
        self.device = device
        self.vae_compression = self.pipe.vae_spatial_compression_ratio
        
        print("✅ LTX pipeline initialized successfully")
    
    
    def round_to_vae_resolution(self, height, width):
        """调整分辨率为 VAE 可接受的值"""
        height = height - (height % self.vae_compression)
        width = width - (width % self.vae_compression)
        return height, width
    
    def generate_video(self, conditioning_frames, prompt, height, width, num_frames, 
                      negative_prompt="worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, temporal inconsistency, jerky motion, choppy framerate",
                      num_inference_steps=30, seed=42, use_upscaler=None, 
                      upscale_denoise_strength=0.4):
        """
        生成视频
        
        Args:
            conditioning_frames: 条件帧列表 (PIL Images 或 numpy arrays)
            prompt: 文本提示
            height: 目标高度
            width: 目标宽度
            num_frames: 生成帧数
            negative_prompt: 负面提示
            num_inference_steps: 推理步数
            seed: 随机种子
            use_upscaler: 是否使用上采样器(默认跟随初始化设置)
            upscale_denoise_strength: 上采样后去噪强度
            
        Returns:
            生成的视频帧列表 (PIL Images)
        """
        if use_upscaler is None:
            use_upscaler = self.use_upscaler
        
        # 转换 conditioning frames 为 LTXVideoCondition
        condition = LTXVideoCondition(video=conditioning_frames, frame_index=0)
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # 如果使用上采样器,先生成较小分辨率
        if use_upscaler and self.use_upscaler:
            downscale_factor = 2 / 3
            downscaled_height = int(height * downscale_factor)
            downscaled_width = int(width * downscale_factor)
            downscaled_height, downscaled_width = self.round_to_vae_resolution(
                downscaled_height, downscaled_width
            )
            
            print(f"  🎨 Generating at {downscaled_width}x{downscaled_height}...")
            
            # Step 1: 生成低分辨率潜变量
            latents = self.pipe(
                conditions=[condition],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=downscaled_width,
                height=downscaled_height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="latent",
            ).frames
            
            # Step 2: 上采样潜变量
            print(f"  📐 Upscaling to {downscaled_width*2}x{downscaled_height*2}...")
            upscaled_latents = self.pipe_upsample(
                latents=latents,
                output_type="latent"
            ).frames
            
            upscaled_height = downscaled_height * 2
            upscaled_width = downscaled_width * 2
            
            # Step 3: 对上采样后的潜变量进行少量去噪
            print(f"  🎨 Refining upscaled video...")
            video = self.pipe(
                conditions=[condition],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=upscaled_width,
                height=upscaled_height,
                num_frames=num_frames,
                denoise_strength=upscale_denoise_strength,
                num_inference_steps=10,
                latents=upscaled_latents,
                decode_timestep=0.05,
                image_cond_noise_scale=0.025,
                generator=generator,
                output_type="pil",
            ).frames[0]
            
            # Step 4: 调整到目标分辨率
            if (upscaled_height != height) or (upscaled_width != width):
                print(f"  📏 Resizing to target {width}x{height}...")
                video = [frame.resize((width, height), Image.LANCZOS) for frame in video]
        
        else:
            # 直接生成目标分辨率
            height, width = self.round_to_vae_resolution(height, width)
            print(f"  🎨 Generating directly at {width}x{height}...")
            
            video = self.pipe(
                conditions=[condition],
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pil",
            ).frames[0]
        
        return video


def compute_relative_pose_correct(rotation_matrices):
    """计算相对位姿(平移 + 偏航角)"""
    T = rotation_matrices.shape[0]
    device = rotation_matrices.device

    poses = torch.zeros(T, 2, device=device)
    yaws = torch.zeros(T, device=device)

    for i in range(1, T):
        T_prev_inv = torch.inverse(rotation_matrices[i-1])
        T_rel = T_prev_inv @ rotation_matrices[i]
        poses[i] = T_rel[:2, 3][:2]
        R_rel = T_rel[:3, :3]
        yaw = torch.atan2(R_rel[1, 0], R_rel[0, 0])
        yaws[i] = yaw

    return poses, yaws


def generate_predictive_prompt(pose, yaw, data_fps, prompt_frames):
    """根据前 n 帧运动生成自然语言 prompt，强调平滑性和连续性"""
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    if isinstance(yaw, torch.Tensor):
        yaw = yaw.cpu().numpy()

    total_forward = float(np.sum(pose[:, 0]))
    total_lateral = float(np.sum(pose[:, 1]))
    net_yaw_change = float(np.sum(yaw))
    abs_yaw_deg = abs(np.degrees(net_yaw_change))

    speeds = np.linalg.norm(pose, axis=1) * data_fps
    avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
    
    # 计算速度变化，用于描述运动稳定性
    speed_std = float(np.std(speeds)) if len(speeds) > 1 else 0.0
    is_steady = speed_std < 1.0  # 根据实际调整阈值

    # 运动趋势描述 - 更自然的语言
    if abs_yaw_deg < 5.0:
        motion_trend = "driving straight ahead"
        turning_desc = "with stable lane keeping"
    elif net_yaw_change > 0:
        if abs_yaw_deg < 15:
            motion_trend = "gently turning left"
            turning_desc = "with smooth steering"
        else:
            motion_trend = "turning left"
            turning_desc = "with controlled steering"
    else:
        if abs_yaw_deg < 15:
            motion_trend = "gently turning right"
            turning_desc = "with smooth steering"
        else:
            motion_trend = "turning right"
            turning_desc = "with controlled steering"

    # 速度描述
    if avg_speed < 5:
        speed_desc = "at low speed"
    elif avg_speed < 15:
        speed_desc = "at moderate speed"
    else:
        speed_desc = "at highway speed"
    
    stability_desc = "steady motion" if is_steady else "gradually changing speed"

    past_seconds = prompt_frames / data_fps
    future_seconds = args.num_frames / data_fps

    # 改进的 prompt：强调连续性、平滑性和真实感
    prompt = (
        f"A high-quality, photorealistic dashboard camera view of autonomous driving. "
        f"Based on the past {past_seconds:.2f} seconds showing {motion_trend} {turning_desc}, "
        f"predict and generate the next {future_seconds:.2f} seconds of realistic driving continuation. "
        f"moving {speed_desc} with {stability_desc}, "
        f"smoothly continue for the next {future_seconds:.2f} seconds. "
        f"Maintain temporal consistency, stable camera perspective, "
        f"natural motion flow without jitter or artifacts, "
        f"clear details, and realistic physics. "
        f"[Technical: forward {total_forward:.2f}m, lateral {total_lateral:.2f}m, "
        f"yaw {np.degrees(net_yaw_change):.1f}°, speed {avg_speed:.2f}m/s]"
    )
    
    return prompt



def tensor_to_pil_frames(frames_tensor):
    """
    将 tensor (T, C, H, W) 转换为 PIL Image 列表
    
    Args:
        frames_tensor: torch.Tensor of shape (T, C, H, W), range [-1, 1] or [0, 1]
    
    Returns:
        List of PIL Images
    """
    pil_frames = []
    for t in range(frames_tensor.shape[0]):
        frame = frames_tensor[t].permute(1, 2, 0).cpu().numpy()
        
        # 归一化到 [0, 1]
        if frame.min() < 0:
            frame = (frame + 1) / 2.0
        frame = np.clip(frame, 0, 1)
        
        # 转换为 [0, 255] uint8
        frame = (frame * 255).astype(np.uint8)
        pil_frames.append(Image.fromarray(frame))
    
    return pil_frames


def pil_frames_to_video(pil_frames, output_path, fps):
    """将 PIL 图像列表保存为 MP4 视频"""
    try:
        export_to_video(pil_frames, output_path, fps=fps)
        return True
    except Exception as e:
        print(f"❌ Error saving video {output_path}: {e}")
        return False


def trim_video_remove_prefix(video_path, output_path, skip_frames, keep_frames, fps):
    """裁剪视频:跳过前 skip_frames 帧,保留接下来的 keep_frames 帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video for trimming: {video_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"❌ Failed to open VideoWriter: {output_path}")
        cap.release()
        return False

    frame_idx = 0
    kept_count = 0
    success = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx >= skip_frames and kept_count < keep_frames:
            frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
            kept_count += 1

        frame_idx += 1

        if kept_count >= keep_frames:
            success = True
            break

    cap.release()
    video_writer.release()

    if success and os.path.exists(output_path):
        cap_check = cv2.VideoCapture(output_path)
        if cap_check.isOpened():
            actual_frame_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_check.release()
            if actual_frame_count == keep_frames:
                return True
    
    print(f"❌ Failed to trim video: {output_path} (expected {keep_frames} frames, got {kept_count})")
    return False


def create_comparison_video(pred_path, gt_path, output_path, fps=8):
    """创建左右对比视频:[Predicted | Ground Truth]"""
    try:
        cap_pred = cv2.VideoCapture(pred_path)
        cap_gt = cv2.VideoCapture(gt_path)

        if not (cap_pred.isOpened() and cap_gt.isOpened()):
            print(f"❌ Cannot open videos for comparison")
            return False

        frame_count_pred = int(cap_pred.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count_pred != frame_count_gt:
            print(f"⚠️ Frame count mismatch: pred={frame_count_pred}, gt={frame_count_gt}")

        width_pred = int(cap_pred.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_pred = int(cap_pred.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_gt = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))

        target_height = max(height_pred, height_gt)
        target_width = width_pred + width_gt
        target_width = target_width if target_width % 2 == 0 else target_width + 1
        target_height = target_height if target_height % 2 == 0 else target_height + 1

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))

        if not video_writer.isOpened():
            print(f"❌ Failed to open VideoWriter")
            return False

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)
        bg_color = (0, 0, 0)

        frame_idx = 0
        while True:
            ret_pred, frame_pred = cap_pred.read()
            ret_gt, frame_gt = cap_gt.read()

            if not (ret_pred and ret_gt):
                break

            frame_pred = cv2.resize(frame_pred, (width_pred, target_height))
            frame_gt = cv2.resize(frame_gt, (width_gt, target_height))

            def add_label(frame, text):
                (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cv2.rectangle(frame, (5, 5), (w + 15, h + 15), bg_color, -1)
                cv2.putText(frame, text, (10, h + 10), font, font_scale, color, thickness)

            add_label(frame_pred, "Predicted")
            add_label(frame_gt, "Ground Truth")

            combined = np.hstack([frame_pred, frame_gt])
            video_writer.write(combined)
            frame_idx += 1

        cap_pred.release()
        cap_gt.release()
        video_writer.release()

        print(f"✅ Comparison video created: {frame_idx} frames")
        return True

    except Exception as e:
        print(f"❌ Error creating comparison: {e}")
        return False


def run_ltx_inference_diffusers(ltx_engine, prompt, conditioning_frames, output_path, args):
    """
    使用 diffusers 库运行 LTX 推理
    
    Args:
        ltx_engine: LTXInferenceEngine 实例
        prompt: 文本提示
        conditioning_frames: PIL Image 列表
        output_path: 输出视频路径
        args: 参数对象
    
    Returns:
        成功则返回输出路径,失败返回 None
    """
    try:
        print(f"  🚀 Running LTX inference with diffusers...")
        
        # 生成视频帧
        video_frames = ltx_engine.generate_video(
            conditioning_frames=conditioning_frames,
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            use_upscaler=args.use_upscaler,
            upscale_denoise_strength=args.upscale_denoise_strength
        )
        
        # 保存为视频
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if pil_frames_to_video(video_frames, output_path, args.frame_rate):
            print(f"  ✅ Video saved: {output_path}")
            return output_path
        else:
            print(f"  ❌ Failed to save video")
            return None
            
    except Exception as e:
        print(f"  ❌ Inference error: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_scene_for_inference(dataset, scene_idx, ltx_engine, args, output_dirs):
    """处理单个 scene 的所有滑动窗口片段"""
    try:
        images, rot_matrices = dataset[scene_idx]
        total_frames = images.shape[0]
        results = []

        prediction_frames = args.num_frames - args.prompt_frames
        required_frames = args.prompt_frames + prediction_frames
        
        if total_frames < required_frames:
            print(f"  ⚠️ Scene {scene_idx}: not enough frames ({total_frames} < {required_frames})")
            return results

        chunk_index = 0
        start_frame = 0

        while start_frame + required_frames <= total_frames:
            cond_end = start_frame + args.prompt_frames
            gt_start = cond_end
            gt_end = gt_start + prediction_frames

            print(f"  📹 Scene {scene_idx}, Window {chunk_index}: "
                  f"cond {start_frame}-{cond_end-1}, gt {gt_start}-{gt_end-1}")

            # 提取 conditioning 数据
            cond_images = images[start_frame:cond_end]
            cond_rots = rot_matrices[start_frame:cond_end]

            # 生成 prompt
            try:
                pose, yaw = compute_relative_pose_correct(cond_rots)
                prompt = generate_predictive_prompt(pose, yaw, 12, args.prompt_frames)
            except Exception as e:
                print(f"    ❌ Prompt generation failed: {e}")
                start_frame += args.window_stride
                chunk_index += 1
                continue

            # --- 1. 保存 conditioning 视频 ---
            cond_name = f"scene_{scene_idx:04d}_window_{chunk_index:03d}_conditioning.mp4"
            cond_path = os.path.join(output_dirs['conditioning'], cond_name)
            
            cond_pil_frames = tensor_to_pil_frames(cond_images)
            if not pil_frames_to_video(cond_pil_frames, cond_path, args.frame_rate):
                print(f"    ❌ Failed to create conditioning video")
                start_frame += args.window_stride
                chunk_index += 1
                continue

            # --- 2. 保存 groundtruth 视频 ---
            gt_images = images[gt_start:gt_end]
            gt_name = f"scene_{scene_idx:04d}_window_{chunk_index:03d}_groundtruth.mp4"
            gt_path = os.path.join(output_dirs['groundtruth'], gt_name)
            
            gt_pil_frames = tensor_to_pil_frames(gt_images)
            if not pil_frames_to_video(gt_pil_frames, gt_path, args.frame_rate):
                print(f"    ❌ Failed to create groundtruth video")
                start_frame += args.window_stride
                chunk_index += 1
                continue

            # --- 3. 保存 prompt 文本 ---
            prompt_name = f"scene_{scene_idx:04d}_window_{chunk_index:03d}_prompt.txt"
            prompt_path = os.path.join(output_dirs['prompts'], prompt_name)
            try:
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt.strip())
                print(f"    💬 Prompt saved: {prompt_path}")
            except Exception as e:
                print(f"    ❌ Failed to save prompt: {e}")
                prompt_path = None

            # --- 4. 运行 LTX 推理 (使用 diffusers) ---
            pred_full_name = f"scene_{scene_idx:04d}_window_{chunk_index:03d}_predicted_full.mp4"
            full_pred_path = os.path.join(output_dirs['predicted'], pred_full_name)
            
            full_pred_path = run_ltx_inference_diffusers(
                ltx_engine, 
                prompt, 
                cond_pil_frames, 
                full_pred_path, 
                args
            )
            
            if not full_pred_path or not os.path.exists(full_pred_path):
                print(f"    ❌ Inference failed for window {chunk_index}")
                start_frame += args.window_stride
                chunk_index += 1
                continue

            # --- 5. 裁剪生成视频:跳过前 prompt_frames,保留后 prediction_frames ---
            pred_trim_name = f"scene_{scene_idx:04d}_window_{chunk_index:03d}_predicted.mp4"
            pred_trim_path = os.path.join(output_dirs['predicted_trimmed'], pred_trim_name)
            
            if not trim_video_remove_prefix(
                full_pred_path, 
                pred_trim_path, 
                skip_frames=args.prompt_frames, 
                keep_frames=prediction_frames, 
                fps=args.frame_rate
            ):
                print(f"    ❌ Failed to trim predicted video")
                start_frame += args.window_stride
                chunk_index += 1
                continue

            # --- 6. 创建对比视频 ---
            comp_name = f"scene_{scene_idx:04d}_window_{chunk_index:03d}_comparison.mp4"
            comp_path = os.path.join(output_dirs['comparisons'], comp_name)
            
            if not create_comparison_video(
                pred_trim_path, 
                gt_path,
                comp_path, 
                args.frame_rate
            ):
                print(f"    ❌ Failed to create comparison video")
            else:
                print(f"    🎥 Comparison video saved: {comp_path}")

            # --- 7. 记录结果 ---
            results.append({
                "scene_idx": scene_idx,
                "window_idx": chunk_index,
                "prompt": prompt,
                "prompt_path": prompt_path,
                "conditioning_video": cond_path,
                "predicted_video_full": full_pred_path,
                "predicted_video": pred_trim_path,
                "groundtruth_video": gt_path,
                "comparison_video": comp_path,
                "conditioning_start_frame": int(start_frame),
                "conditioning_end_frame": int(cond_end - 1),
                "prediction_start_frame": int(gt_start),
                "prediction_end_frame": int(gt_end - 1),
                "prediction_duration_seconds": prediction_frames / args.frame_rate
            })

            start_frame += args.window_stride
            chunk_index += 1

        return results

    except Exception as e:
        print(f"  ❌ Error processing scene {scene_idx}: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_results(all_results, output_dir):
    """保存结果为 CSV 和 JSON"""
    csv_path = os.path.join(output_dir, "inference_results.csv")
    json_path = os.path.join(output_dir, "inference_results.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Results saved to:")
    print(f"   - {csv_path}")
    print(f"   - {json_path}")


def main():
    parser = argparse.ArgumentParser(description="LTX Video Prediction Inference with Diffusers (Sliding Window)")

    # 数据集路径
    parser.add_argument("--nuscenes_root", type=str, required=True, help="Path to nuScenes dataset root")
    parser.add_argument("--nuscenes_train_json", type=str, required=True, help="Path to train split JSON")

    # 输出路径
    parser.add_argument("--output_dir", type=str, default="./ltx_inference_results", help="Root output directory")

    # 滑动窗口参数
    parser.add_argument("--prompt_frames", type=int, default=6, help="Number of conditioning frames")
    parser.add_argument("--window_stride", type=int, default=8, help="Stride between windows in frames")

    # LTX 模型参数
    parser.add_argument("--model_path", type=str, default="Lightricks/LTX-Video-0.9.7-dev", 
                       help="HuggingFace model ID or local path")
    parser.add_argument("--upscaler_path", type=str, default="Lightricks/ltxv-spatial-upscaler-0.9.7",
                       help="Spatial upscaler model ID or local path ")
    parser.add_argument("--height", type=int, default=704, help="Output video height")
    parser.add_argument("--width", type=int, default=1280, help="Output video width")
    parser.add_argument("--num_frames", type=int, default=41, help="Total number of frames to generate (including conditioning)")
    parser.add_argument("--frame_rate", type=int, default=8, help="Frame rate for video I/O")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion inference steps")
    parser.add_argument("--use_upscaler", action="store_true", help="Use spatial upsampler for higher quality")
    parser.add_argument("--upscale_denoise_strength", type=float, default=0.4, 
                       help="Denoise strength after upscaling (0.0-1.0)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")

    # 处理范围
    parser.add_argument("--start_scene", type=int, default=0, help="Start from which scene index")
    parser.add_argument("--num_scenes", type=int, default=10, help="Number of scenes to process (-1 for all)")

    global args
    args = parser.parse_args()

    print("=" * 80)
    print("LTX VIDEO PREDICTION INFERENCE WITH DIFFUSERS (SLIDING WINDOW)")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"Conditioning: {args.prompt_frames} frames ({args.prompt_frames / args.frame_rate:.2f}s)")
    print(f"Prediction: {args.num_frames} frames ({args.num_frames / args.frame_rate:.2f}s)")
    print(f"Resolution: {args.width}x{args.height} @ {args.frame_rate}fps")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Use upscaler: {args.use_upscaler}")
    print(f"Sliding window stride: {args.window_stride} frames")

    # 创建输出目录
    output_dirs = {
        'conditioning': os.path.join(args.output_dir, "conditioning_videos"),
        'predicted': os.path.join(args.output_dir, "predicted_videos_full"),
        'predicted_trimmed': os.path.join(args.output_dir, "predicted_videos"),
        'groundtruth': os.path.join(args.output_dir, "groundtruth_videos"),
        'prompts': os.path.join(args.output_dir, "prompts"),
        'comparisons': os.path.join(args.output_dir, "comparisons"),
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    # 初始化 LTX 推理引擎
    ltx_engine = LTXInferenceEngine(
        model_path=args.model_path,
        device=args.device,
        use_upscaler=args.use_upscaler,
        upscaler_path=args.upscaler_path
    )

    # 加载数据集
    print("\n📂 Loading dataset...")
    dataset = TrainDataset(
        args.nuscenes_root,
        args.nuscenes_train_json,
        condition_frames=200,
        downsample_fps=args.frame_rate,
        h=args.height,
        w=args.width
    )
    total_scenes = len(dataset)
    print(f"✓ Loaded {total_scenes} scenes")

    # 确定处理范围
    start = args.start_scene
    end = total_scenes if args.num_scenes == -1 else min(start + args.num_scenes, total_scenes)
    scene_indices = list(range(start, end))

    print(f"🎬 Processing scenes: {start} to {end-1} ({len(scene_indices)} total)")

    # 主循环
    all_results = []
    for scene_idx in tqdm(scene_indices, desc="Processing Scenes"):
        scene_results = process_scene_for_inference(dataset, scene_idx, ltx_engine, args, output_dirs)
        all_results.extend(scene_results)
        
        # 定期保存中间结果
        if len(all_results) > 0 and len(all_results) % 10 == 0:
            save_results(all_results, args.output_dir)
            print(f"💾 Intermediate results saved ({len(all_results)} videos processed)")

    # 保存最终结果
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total videos processed: {len(all_results)}")
    if all_results:
        save_results(all_results, args.output_dir)
        print(f"✅ All outputs saved to: {args.output_dir}")
    else:
        print("⚠️ No videos were successfully generated.")


if __name__ == "__main__":
    main()




