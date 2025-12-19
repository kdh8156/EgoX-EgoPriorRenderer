# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import socket
import time
from typing import Tuple, List, Optional

from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import viser
import viser.transforms as tf

from matplotlib import cm
from PIL import Image
from rich.logging import RichHandler
import pandas as pd
from scipy.spatial.transform import Rotation as R

try:
    import projectaria_tools.core.mps as mps
    from projectaria_tools.core.mps.utils import filter_points_from_confidence
    PROJECTARIA_AVAILABLE = True
except ImportError:
    PROJECTARIA_AVAILABLE = False
    print("Warning: projectaria-tools not available. Semi-dense point cloud visualization will be disabled.")

from vipe.utils.cameras import CameraType
from vipe.utils.depth import reliable_depth_mask_range
from vipe.utils.io import (
    ArtifactPath,
    read_depth_artifacts,
    read_instance_artifacts,
    read_intrinsics_artifacts,
    read_pose_artifacts,
    read_rgb_artifacts,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


def compute_robust_mean_tensors(stacked_depth: torch.Tensor, stacked_rgb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU 벡터화된 버전: 픽셀별 NaN-aware 평균을 병렬로 계산.
    """
    logger.info("Computing mean background using nanmean with GPU vectorization...")
    
    # NaN을 무시하고 평균 계산
    robust_depth = torch.nanmean(stacked_depth, dim=0)
    robust_rgb = torch.nanmean(stacked_rgb, dim=0)
    
    # NaN 값들을 0으로 대체
    robust_depth = torch.nan_to_num(robust_depth, nan=0.0)
    robust_rgb = torch.nan_to_num(robust_rgb, nan=0.0)
    
    return robust_depth, robust_rgb


@dataclass
class GlobalContext:
    artifacts: list[ArtifactPath]
    use_mean_bg: bool = False
    use_exo_intrinsic_gt: bool = False
    mean_depth: torch.Tensor = None
    mean_rgb: torch.Tensor = None
    take_uuid: str = None
    start_frame: int = None
    ego_extrinsics_list: List[np.ndarray] = None
    exo_cam_to_world: np.ndarray = None
    semidense_points: np.ndarray = None  # Ego4D semi-dense point cloud
    gt_exo_intrinsics_K: np.ndarray | None = None  # 3x3
    gt_exo_image_size: Tuple[int, int] | None = None  # (width, height)
    ego_manual: bool = False  # Manual ego camera control mode
    
    def __post_init__(self):
        # dataclass에서 None으로 초기화된 tensor 필드들 처리
        if self.mean_depth is None:
            self.mean_depth = None
        if self.mean_rgb is None:
            self.mean_rgb = None


_global_context: GlobalContext | None = None


@dataclass
class SceneFrameHandle:
    frame_handle: viser.FrameHandle
    frustum_handle: viser.CameraFrustumHandle
    pcd_handle: viser.PointCloudHandle | None = None
    dynamic_pcd_handle: viser.PointCloudHandle | None = None

    def __post_init__(self):
        self.visible = False

    @property
    def visible(self) -> bool:
        return self.frame_handle.visible

    @visible.setter
    def visible(self, value: bool):
        self.frame_handle.visible = value
        self.frustum_handle.visible = value
        if self.pcd_handle is not None:
            self.pcd_handle.visible = value
        # Dynamic point cloud는 추가적으로 show_dynamic_objects 설정도 고려
        if self.dynamic_pcd_handle is not None:
            # ClientClosures instance에 접근하기 위해 여기서는 단순히 value만 사용
            # GUI 상태는 GUI 컨트롤의 on_update에서 처리
            self.dynamic_pcd_handle.visible = value


def compute_mean_background_depth(artifact_path: ArtifactPath, 
                                  spatial_subsample: int = 2, max_frames: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    모든 프레임에서 background depth와 RGB의 nanmean을 미리 계산.
    """
    logger.info("Computing mean background depth using nanmean...")
    
    # GPU device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def none_it(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    def none_it_mask(inner_it):
        try:
            for item in inner_it:
                yield item
        except FileNotFoundError:
            while True:
                yield None, None

    valid_depth_frames = []
    valid_rgb_frames = []
    frame_count = 0

    for frame_idx, ((_, rgb), (_, depth), (_, instance_mask)) in enumerate(
        zip(
            read_rgb_artifacts(artifact_path.rgb_path),
            none_it(read_depth_artifacts(artifact_path.depth_path)),
            none_it_mask(read_instance_artifacts(artifact_path.mask_path)),
        )
    ):
        if depth is None:
            continue
        
        if max_frames and frame_count >= max_frames:
            break

        # GPU로 이동
        rgb = rgb.to(device)
        depth = depth.to(device)
        if instance_mask is not None:
            instance_mask = instance_mask.to(device)

        # Background만 유지 (instance_id == 0)
        if instance_mask is not None:
            static_mask = (instance_mask == 0)
        else:
            static_mask = torch.ones_like(depth, dtype=torch.bool, device=device)
        
        # 유효한 depth 마스크
        depth_mask = reliable_depth_mask_range(depth)
        final_mask = static_mask & depth_mask

        # 무효한 픽셀을 NaN으로 설정
        masked_depth = depth.clone().float()
        masked_depth[~final_mask] = float('nan')
        
        masked_rgb = rgb.clone().float()
        masked_rgb[~final_mask.unsqueeze(-1).expand_as(rgb)] = float('nan')

        # Spatial subsampling 적용
        if spatial_subsample > 1:
            masked_depth = masked_depth[::spatial_subsample, ::spatial_subsample]
            masked_rgb = masked_rgb[::spatial_subsample, ::spatial_subsample]

        valid_depth_frames.append(masked_depth)
        valid_rgb_frames.append(masked_rgb)
        
        frame_count += 1
        if frame_idx % 10 == 0:
            logger.info(f"Processed frame {frame_idx} for mean background computation.")

    if not valid_depth_frames:
        logger.warning("No valid background frames found.")
        return None, None

    logger.info(f"Computing nanmean from {len(valid_depth_frames)} frames...")
    
    stacked_depth = torch.stack(valid_depth_frames, dim=0)  # [N_frames, H, W]
    stacked_rgb = torch.stack(valid_rgb_frames, dim=0)      # [N_frames, H, W, 3]
    
    # Nanmean 계산
    mean_depth, mean_rgb = compute_robust_mean_tensors(stacked_depth, stacked_rgb)
    
    logger.info(f"Mean background depth computed: {mean_depth.shape}, device: {mean_depth.device}")
    return mean_depth, mean_rgb


def build_dynamic_points_for_frame(artifact_path: ArtifactPath, frame_idx: int, c2w: np.ndarray, 
                                   rgb: torch.Tensor, depth: torch.Tensor, instance_mask: torch.Tensor,
                                   camera_type: CameraType, intr: torch.Tensor, spatial_subsample: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    특정 프레임에서 dynamic objects (instance_id != 0)의 point cloud를 생성.
    
    Args:
        artifact_path: ViPE artifact path
        frame_idx: Frame index
        c2w: Camera-to-world transformation matrix
        rgb: RGB image tensor
        depth: Depth tensor
        instance_mask: Instance segmentation mask
        camera_type: Camera type
        intr: Camera intrinsics
        spatial_subsample: Spatial subsampling factor
        
    Returns:
        dynamic_points: Nx3 array of 3D points (camera coordinates)
        dynamic_colors: Nx3 array of RGB colors (0-255)
    """
    if depth is None or instance_mask is None:
        return np.empty((0, 3)), np.empty((0, 3))

    frame_height, frame_width = rgb.shape[:2]

    # RGB 처리
    sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
    sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

    # Ray 생성
    camera_model = camera_type.build_camera_model(intr)
    disp_v, disp_u = torch.meshgrid(
        torch.arange(frame_height).float()[::spatial_subsample],
        torch.arange(frame_width).float()[::spatial_subsample],
        indexing="ij",
    )
    if camera_type == CameraType.PANORAMA:
        disp_v = disp_v / (frame_height - 1)
        disp_u = disp_u / (frame_width - 1)
    disp = torch.ones_like(disp_v)
    pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
    rays = pts[..., :3].numpy()
    if camera_type != CameraType.PANORAMA:
        rays /= rays[..., 2:3]

    # Point cloud 생성 (카메라 좌표계)
    pcd_camera = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]
    
    # Depth mask 적용
    depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()

    # Dynamic objects만 유지 (instance_id != 0)
    instance_mask_np = instance_mask.cpu().numpy() if hasattr(instance_mask, 'cpu') else instance_mask
    dynamic_mask = (instance_mask_np != 0)
    dynamic_mask_sub = dynamic_mask[::spatial_subsample, ::spatial_subsample]
    final_mask = depth_mask & dynamic_mask_sub

    # Flatten and filter
    pcd_flat = pcd_camera.reshape(-1, 3)
    rgb_flat = sampled_rgb.reshape(-1, 3)
    mask_flat = final_mask.reshape(-1)

    valid_points = pcd_flat[mask_flat]
    valid_colors = rgb_flat[mask_flat]

    logger.info(f"Frame {frame_idx}: Generated {len(valid_points)} dynamic points")
    
    return valid_points, valid_colors


class ClientClosures:
    """
    All class methods automatically capture 'self', ensuring proper locals.
    """

    def __init__(self, client: viser.ClientHandle):
        self.client = client

        async def _run():
            try:
                await self.run()
            except asyncio.CancelledError:
                pass
            finally:
                self.cleanup()

        # Don't await to not block the rest of the coroutine.
        self.task = asyncio.create_task(_run())

        self.gui_playback_handle: viser.GuiFolderHandle | None = None
        self.gui_timestep: viser.GuiSliderHandle | None = None
        self.gui_framerate: viser.GuiSliderHandle | None = None
        self.scene_frame_handles: list[SceneFrameHandle] = []
        self.current_displayed_timestep: int = 0

        # added
        self.gui_playing: viser.GuiCheckboxHandle | None = None
        self.gui_show_all_frames: viser.GuiCheckboxHandle | None = None
        self.ego_frustum_handles: list[viser.CameraFrustumHandle] = []
        self.ego_trajectory_handles: list = []  # 점선을 위한 여러 spline handles
        self.semidense_pcd_handle: viser.PointCloudHandle | None = None
        
        # Manual ego camera control
        self.manual_ego_frame: viser.FrameHandle | None = None
        self.manual_ego_frustum: viser.CameraFrustumHandle | None = None
        self.manual_ego_transform_handle: viser.TransformControlsHandle | None = None
        self.gui_manual_extrinsic_text: viser.GuiMarkdownHandle | None = None

    async def stop(self):
        self.task.cancel()
        await self.task

    async def run(self):
        logger.info(f"Client {self.client.client_id} connected")

        all_artifacts = self.global_context().artifacts

        with self.client.gui.add_folder("Sample"):
            self.gui_id = self.client.gui.add_slider(
                "Artifact ID", min=0, max=len(all_artifacts) - 1, step=1, initial_value=0
            )
            gui_id_changer = self.client.gui.add_button_group(label="ID +/-", options=["Prev", "Next"])

            @gui_id_changer.on_click
            async def _(_) -> None:
                if gui_id_changer.value == "Prev":
                    self.gui_id.value = (self.gui_id.value - 1) % len(all_artifacts)
                else:
                    self.gui_id.value = (self.gui_id.value + 1) % len(all_artifacts)

            self.gui_name = self.client.gui.add_text("Artifact Name", "")
            self.gui_t_sub = self.client.gui.add_slider("Temporal subsample", min=1, max=16, step=1, initial_value=1)
            self.gui_s_sub = self.client.gui.add_slider("Spatial subsample", min=1, max=8, step=1, initial_value=2)
            self.gui_id.on_update(self.on_sample_update)
            self.gui_t_sub.on_update(self.on_sample_update)
            self.gui_s_sub.on_update(self.on_sample_update)

        with self.client.gui.add_folder("Scene"):
            self.gui_point_size = self.client.gui.add_slider(
                "Point size", min=0.0001, max=0.1, step=0.001, initial_value=0.001
            )

            # Update point cloud size
            @self.gui_point_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    if frame_node.pcd_handle is not None:
                        frame_node.pcd_handle.point_size = self.gui_point_size.value
                    if frame_node.dynamic_pcd_handle is not None:
                        frame_node.dynamic_pcd_handle.point_size = self.gui_point_size.value * 1.2

            self.gui_frustum_size = self.client.gui.add_slider(
                "Frustum size", min=0.01, max=0.5, step=0.01, initial_value=0.15
            )

            @self.gui_frustum_size.on_update
            async def _(_) -> None:
                for frame_node in self.scene_frame_handles:
                    frame_node.frustum_handle.scale = self.gui_frustum_size.value

            self.gui_colorful_frustum_toggle = self.client.gui.add_checkbox(
                "Colorful Frustum",
                initial_value=False,
            )

            @self.gui_colorful_frustum_toggle.on_update
            async def _(_) -> None:
                self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

            # 동적 객체 필터링 컨트롤 추가
            self.gui_filter_dynamic_objects = self.client.gui.add_checkbox(
                "Filter Dynamic Objects",
                initial_value=True,
                hint="Remove dynamic objects (person, car, etc.) from point cloud"
            )

            @self.gui_filter_dynamic_objects.on_update
            async def _(_) -> None:
                await self.on_sample_update(None)

            # Dynamic objects 표시 컨트롤 추가
            self.gui_show_dynamic_objects = self.client.gui.add_checkbox(
                "Show Dynamic Objects",
                initial_value=True,
                hint="Show dynamic objects (people, cars, etc.) as separate point cloud"
            )

            @self.gui_show_dynamic_objects.on_update
            async def _(_) -> None:
                # Dynamic point clouds의 visibility 토글
                for frame_node in self.scene_frame_handles:
                    if frame_node.dynamic_pcd_handle is not None:
                        frame_node.dynamic_pcd_handle.visible = self.gui_show_dynamic_objects.value and frame_node.visible

            # Semi-dense point cloud 표시 컨트롤 추가
            self.gui_show_semidense_pcd = self.client.gui.add_checkbox(
                "Show Semi-dense Points",
                initial_value=True,
                hint="Show Ego4D semi-dense point cloud (pseudo GT)"
            )

            @self.gui_show_semidense_pcd.on_update
            async def _(_) -> None:
                # Semi-dense point cloud visibility 토글
                if hasattr(self, 'semidense_pcd_handle') and self.semidense_pcd_handle is not None:
                    self.semidense_pcd_handle.visible = self.gui_show_semidense_pcd.value

            self.gui_fov = self.client.gui.add_slider("FoV", min=30.0, max=120.0, step=1.0, initial_value=60.0)

            @self.gui_fov.on_update
            async def _(_) -> None:
                self.client.camera.fov = np.deg2rad(self.gui_fov.value)

            gui_snapshot = self.client.gui.add_button(
                "Snapshot",
                hint="Take a snapshot of the current scene",
            )

            # Async get_render does not work at the moment, we will put into thread loop.
            @gui_snapshot.on_click
            def _(_) -> None:
                current_artifact = self.global_context().artifacts[self.gui_id.value]
                file_name = f"{current_artifact.base_path.name}_{current_artifact.artifact_name}.png"
                snapshot_img = self.client.get_render(height=720, width=1280, transport_format="png")
                self.client.send_file_download(file_name, iio.imwrite("<bytes>", snapshot_img, extension=".png"))

        # Manual Ego Camera Control
        if self.global_context().ego_manual:
            with self.client.gui.add_folder("Manual Ego Camera"):
                gui_show_manual_ego = self.client.gui.add_checkbox(
                    "Show Manual Camera",
                    initial_value=True,
                    hint="Show/hide manual ego camera frustum"
                )
                
                gui_reset_manual_ego = self.client.gui.add_button(
                    "Reset to Default Pose",
                    hint="Reset manual camera to 90° CCW rotation"
                )
                
                self.gui_manual_extrinsic_text = self.client.gui.add_markdown(
                    "**World to Manual Cam Extrinsic (4x3):**\n```\nInitializing...\n```"
                )
                
                @gui_show_manual_ego.on_update
                async def _(_) -> None:
                    if self.manual_ego_frame is not None:
                        self.manual_ego_frame.visible = gui_show_manual_ego.value
                    if self.manual_ego_frustum is not None:
                        self.manual_ego_frustum.visible = gui_show_manual_ego.value
                    if self.manual_ego_transform_handle is not None:
                        self.manual_ego_transform_handle.visible = gui_show_manual_ego.value
                
                @gui_reset_manual_ego.on_click
                def _(_) -> None:
                    self._reset_manual_ego_camera()

        await self.on_sample_update(None)

        while True:
            if (self.gui_playing is not None and self.gui_playing.value # added
                and self.gui_framerate is not None and self.gui_framerate.value > 0
            ):
                self._incr_timestep()
                await asyncio.sleep(1.0 / self.gui_framerate.value)
            else:
                await asyncio.sleep(1.0)

    async def on_sample_update(self, _):
        with self.client.atomic():
            self._rebuild_scene()
            
            # Add ego camera frustums and world axes if take_uuid is provided
            take_uuid = self.global_context().take_uuid
            if take_uuid is not None:
                self._add_ego_camera_frustums()
                self._add_ego_trajectory()  # Add trajectory visualization
                self._add_ego4d_world_axes()
                self._add_semidense_point_cloud()
            
            # Add manual ego camera if enabled
            if self.global_context().ego_manual:
                self._add_manual_ego_camera()
                
        self._rebuild_playback_gui()
        self._set_frustum_color(self.gui_colorful_frustum_toggle.value)

    def _set_frustum_color(self, colorful: bool):
        for frame_idx, frame_node in enumerate(self.scene_frame_handles):
            if not colorful:
                frame_node.frustum_handle.color = (0, 0, 0)
            else:
                # Use a rainbow color based on the frame index
                denom = len(self.scene_frame_handles) - 1
                rainbow_value = cm.jet(1.0 - frame_idx / denom)[:3]
                rainbow_value = tuple((int(c * 255) for c in rainbow_value))
                frame_node.frustum_handle.color = rainbow_value

    def _add_ego_camera_frustums(self):
        """Add ego camera frustums to the scene in exo camera coordinate system."""
        logger.info("Adding ego camera frustums to scene...")
        
        ego_extrinsics_list = self.global_context().ego_extrinsics_list
        exo_cam_to_world = self.global_context().exo_cam_to_world
        
        if ego_extrinsics_list is None or exo_cam_to_world is None:
            logger.warning("Missing ego camera data, skipping ego frustum visualization")
            return

        # Compute transformation from Ego4D world to ViPE world using exo camera alignment
        try:
            current_artifact = self.global_context().artifacts[self.gui_id.value]
            # Use first ViPE exo camera pose as world alignment reference
            vipe_c2w_iter = read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy()
            vipe_exo_c2w = next(iter(vipe_c2w_iter))
            ego_exo_c2w = exo_cam_to_world
            # T_egoWorld_to_vipeWorld maps Ego4D world coords -> ViPE world coords
            T_egoW_to_vipeW = vipe_exo_c2w @ np.linalg.inv(ego_exo_c2w)
            logger.info("Computed Ego4D->ViPE world alignment from exo camera poses")
        except Exception as e:
            logger.warning(f"Failed to compute Ego4D->ViPE alignment: {e}")
            return

        # Use default FOV for ego camera frustums (no intrinsics needed)
        fov = np.radians(60.0)  # Default 60 degrees FOV
        logger.info(f"Using default ego camera FOV: {np.degrees(fov):.1f}°")
        
        # Add ego camera frustums for each frame (placed in ViPE world coordinates)
        self.ego_frustum_handles = []
        for frame_idx, ego_w2c in enumerate(ego_extrinsics_list):
            # Convert ego W2C (3x4) to 4x4 matrix
            ego_w2c_4x4 = np.eye(4)
            ego_w2c_4x4[:3, :] = ego_w2c
            
            # Convert to ego C2W in Ego4D world
            ego_c2w_egoW = np.linalg.inv(ego_w2c_4x4)

            # Convert ego camera pose into ViPE world coordinates
            ego_c2w_vipeW = T_egoW_to_vipeW @ ego_c2w_egoW
            
            # Extract position and rotation
            ego_position = ego_c2w_vipeW[:3, 3]
            ego_rotation = ego_c2w_vipeW[:3, :3]
            
            # Convert rotation matrix to quaternion for viser
            ego_quat = tf.SO3.from_matrix(ego_rotation).wxyz
            
            # Add ego camera frustum
            ego_frustum_handle = self.client.scene.add_camera_frustum(
                f"/ego_cameras/frame_{frame_idx}",
                fov=fov,
                aspect=1.0,  # Assume square aspect ratio
                scale=self.gui_frustum_size.value * 0.8,  # Slightly smaller than exo frustums
                wxyz=ego_quat,
                position=ego_position,
            )
            
            # Color ego frustums differently (green-ish)
            ego_frustum_handle.color = (0, 255, 0)  # Green color for ego cameras
            
            # Initially hide all ego frustums (will be shown based on timeline)
            ego_frustum_handle.visible = False
            
            self.ego_frustum_handles.append(ego_frustum_handle)
            logger.info(f"Added ego camera frustum for frame {frame_idx} at position {ego_position}")
        
        logger.info(f"Successfully added {len(ego_extrinsics_list)} ego camera frustums")

    def _add_ego_trajectory(self):
        """Add trajectory line connecting ego camera positions."""
        logger.info("Adding ego camera trajectory...")
        
        ego_extrinsics_list = self.global_context().ego_extrinsics_list
        exo_cam_to_world = self.global_context().exo_cam_to_world
        
        if ego_extrinsics_list is None or exo_cam_to_world is None:
            logger.warning("Missing ego camera data, skipping trajectory visualization")
            return
        
        try:
            current_artifact = self.global_context().artifacts[self.gui_id.value]
            # Use first ViPE exo camera pose as world alignment reference
            vipe_c2w_iter = read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy()
            vipe_exo_c2w = next(iter(vipe_c2w_iter))
            ego_exo_c2w = exo_cam_to_world
            # T_egoWorld_to_vipeWorld maps Ego4D world coords -> ViPE world coords
            T_egoW_to_vipeW = vipe_exo_c2w @ np.linalg.inv(ego_exo_c2w)
        except Exception as e:
            logger.warning(f"Failed to compute Ego4D->ViPE alignment for trajectory: {e}")
            return
        
        # Extract ego camera positions in ViPE world coordinates
        trajectory_points = []
        for ego_w2c in ego_extrinsics_list:
            # Convert ego W2C (3x4) to 4x4 matrix
            ego_w2c_4x4 = np.eye(4)
            ego_w2c_4x4[:3, :] = ego_w2c
            
            # Convert to ego C2W in Ego4D world
            ego_c2w_egoW = np.linalg.inv(ego_w2c_4x4)
            
            # Convert ego camera pose into ViPE world coordinates
            ego_c2w_vipeW = T_egoW_to_vipeW @ ego_c2w_egoW
            
            # Extract position
            ego_position = ego_c2w_vipeW[:3, 3]
            trajectory_points.append(ego_position)
        
        trajectory_points = np.array(trajectory_points)
        
        # Add trajectory as thick dashed line (점선 효과)
        # 점선을 만들기 위해 일정 간격으로 선분을 건너뛰며 추가
        if len(trajectory_points) >= 2:
            # 점선 효과: 2개씩 건너뛰며 선분 생성 (dash pattern)
            dash_segments = []
            for i in range(0, len(trajectory_points) - 1, 3):  # 3칸마다 1개씩 그림
                if i + 1 < len(trajectory_points):
                    # 각 dash는 2-3개 포인트로 구성
                    end_idx = min(i + 2, len(trajectory_points))
                    dash_segments.append(trajectory_points[i:end_idx])
            
            # 여러 개의 spline을 생성해서 점선 효과
            self.ego_trajectory_handles = []
            for idx, dash_points in enumerate(dash_segments):
                if len(dash_points) >= 2:
                    handle = self.client.scene.add_spline_catmull_rom(
                        f"/ego_trajectory/dash_{idx}",
                        positions=dash_points,
                        line_width=5.0,  # 두꺼운 선 (2.0 -> 5.0)
                        color=(255, 0, 0),  # 빨간색
                        segments=20,  # 각 dash의 부드러움
                    )
                    self.ego_trajectory_handles.append(handle)
            
            logger.info(f"Added dashed ego trajectory with {len(dash_segments)} segments from {len(trajectory_points)} points")
        else:
            logger.warning("Not enough points to create trajectory")

    def _add_ego4d_world_axes(self):
        """Add Ego4D world coordinate system axes to the scene."""
        logger.info("Adding Ego4D world coordinate system axes...")
        
        ego_extrinsics_list = self.global_context().ego_extrinsics_list
        exo_cam_to_world = self.global_context().exo_cam_to_world
        
        if ego_extrinsics_list is None or exo_cam_to_world is None:
            logger.warning("Missing ego camera data, skipping Ego4D world axes visualization")
            return

        # Compute transformation from Ego4D world to ViPE world using exo camera alignment
        try:
            current_artifact = self.global_context().artifacts[self.gui_id.value]
            # Use first ViPE exo camera pose as world alignment reference
            vipe_c2w_iter = read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy()
            vipe_exo_c2w = next(iter(vipe_c2w_iter))
            ego_exo_c2w = exo_cam_to_world
            # T_egoWorld_to_vipeWorld maps Ego4D world coords -> ViPE world coords
            T_egoW_to_vipeW = vipe_exo_c2w @ np.linalg.inv(ego_exo_c2w)
            logger.info("Computed Ego4D->ViPE world alignment from exo camera poses")
        except Exception as e:
            logger.warning(f"Failed to compute Ego4D->ViPE alignment: {e}")
            return
        
        # Ego4D world origin in ViPE world coordinates
        ego_world_origin_vipe = T_egoW_to_vipeW[:3, 3]
        
        # Add Ego4D world coordinate frame
        ego_world_frame = self.client.scene.add_frame(
            "/ego4d_world/frame",
            axes_length=0.05,  # Length of coordinate axes
            axes_radius=0.005,  # Thickness of coordinate axes
            origin_radius=0.005,  # Size of origin sphere
        )
        
        # Set the frame position and orientation using the transformation matrix
        ego_world_frame.wxyz = tf.SO3.from_matrix(T_egoW_to_vipeW[:3, :3]).wxyz
        ego_world_frame.position = ego_world_origin_vipe
        
        logger.info(f"Added Ego4D world coordinate system at origin: {ego_world_origin_vipe}")

    def _add_semidense_point_cloud(self):
        """Add Ego4D semi-dense point cloud to the scene."""
        logger.info("Adding Ego4D semi-dense point cloud...")
        
        semidense_points = self.global_context().semidense_points
        ego_extrinsics_list = self.global_context().ego_extrinsics_list
        exo_cam_to_world = self.global_context().exo_cam_to_world
        
        if semidense_points is None:
            logger.warning("No semi-dense points available, skipping visualization")
            return
            
        if ego_extrinsics_list is None or exo_cam_to_world is None:
            logger.warning("Missing ego camera data, using semi-dense points in original coordinates")
            # Use points as-is (already in Ego4D world coordinates)
            points_vipe = semidense_points
        else:
            # Transform semi-dense points from Ego4D world to ViPE world coordinates
            try:
                current_artifact = self.global_context().artifacts[self.gui_id.value]
                # Use first ViPE exo camera pose as world alignment reference
                vipe_c2w_iter = read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy()
                vipe_exo_c2w = next(iter(vipe_c2w_iter))
                ego_exo_c2w = exo_cam_to_world
                # T_egoWorld_to_vipeWorld maps Ego4D world coords -> ViPE world coords
                T_egoW_to_vipeW = vipe_exo_c2w @ np.linalg.inv(ego_exo_c2w)
                
                # Transform points to ViPE world coordinates
                # Add homogeneous coordinate
                points_homo = np.hstack([semidense_points, np.ones((semidense_points.shape[0], 1))])
                # Transform points
                points_vipe_homo = (T_egoW_to_vipeW @ points_homo.T).T
                points_vipe = points_vipe_homo[:, :3]
                
                logger.info("Transformed semi-dense points to ViPE world coordinates")
            except Exception as e:
                logger.warning(f"Failed to transform semi-dense points: {e}")
                points_vipe = semidense_points
        
        # Create colors for the points (use a consistent, pale orange color)
        colors = np.full((points_vipe.shape[0], 3), [0.9, 0.7, 0.5])
        
        # Add point cloud to scene
        self.semidense_pcd_handle = self.client.scene.add_point_cloud(
            "/semidense_points",
            points=points_vipe,
            colors=colors,
            point_size=0.005,  # Smaller point size (reduced from 0.01 to 0.005)
        )
        
        # Set visibility based on GUI control
        if hasattr(self, 'gui_show_semidense_pcd') and self.gui_show_semidense_pcd is not None:
            self.semidense_pcd_handle.visible = self.gui_show_semidense_pcd.value
        else:
            self.semidense_pcd_handle.visible = True
        
        logger.info(f"Added {points_vipe.shape[0]} semi-dense points to scene")

    def _add_manual_ego_camera(self):
        """Add manual ego camera with transform controls."""
        logger.info("Adding manual ego camera...")
        
        default_rotation = R.from_euler('z', 90, degrees=True).as_matrix()
        default_position = np.array([0.0, 0.0, 0.0])
        default_quat = tf.SO3.from_matrix(default_rotation).wxyz
        
        # Create transform controls
        self.manual_ego_transform_handle = self.client.scene.add_transform_controls(
            "/manual_ego_camera/controls",
            scale=0.5,
            line_width=5.0,
        )
        self.manual_ego_transform_handle.wxyz = default_quat
        self.manual_ego_transform_handle.position = default_position
        
        # Create camera frustum
        fov = np.radians(60.0)
        self.manual_ego_frustum = self.client.scene.add_camera_frustum(
            "/manual_ego_camera/frustum",
            fov=fov,
            aspect=1.0,
            scale=0.2,
            color=(255, 165, 0),
            wxyz=default_quat,
            position=default_position,
        )
        
        # Create frame handle
        self.manual_ego_frame = self.client.scene.add_frame(
            "/manual_ego_camera/frame",
            wxyz=default_quat,
            position=default_position,
            axes_length=0.2,
            axes_radius=0.015,
        )
        
        # Update callback for transform controls
        @self.manual_ego_transform_handle.on_update
        def _(_) -> None:
            self._update_manual_ego_camera()
        
        # Initial update
        self._update_manual_ego_camera()
        
        logger.info("Manual ego camera added")
    
    def _reset_manual_ego_camera(self):
        """Reset manual ego camera to default pose (90° CCW around Z)."""
        if self.manual_ego_transform_handle is None:
            return
            
        # Default pose: 90° counter-clockwise rotation around Z-axis
        default_rotation = R.from_euler('z', 90, degrees=True).as_matrix()
        default_position = np.array([0.0, 0.0, 0.0])
        default_quat = tf.SO3.from_matrix(default_rotation).wxyz
        
        # Update transform control
        self.manual_ego_transform_handle.wxyz = default_quat
        self.manual_ego_transform_handle.position = default_position
        
        # Update will be triggered by on_update callback
        logger.info("Manual ego camera reset to default pose")
    
    def _update_manual_ego_camera(self):
        """Update manual ego camera frustum and extrinsic matrix display."""
        if self.manual_ego_transform_handle is None:
            return
        
        # Get current transform
        position = self.manual_ego_transform_handle.position
        wxyz = self.manual_ego_transform_handle.wxyz
        
        # Update frustum to match transform control
        if self.manual_ego_frustum is not None:
            self.manual_ego_frustum.position = position
            self.manual_ego_frustum.wxyz = wxyz
        
        # Update frame to match transform control
        if self.manual_ego_frame is not None:
            self.manual_ego_frame.position = position
            self.manual_ego_frame.wxyz = wxyz
        
        # Compute camera-to-world matrix (C2W)
        rotation_matrix = tf.SO3(wxyz).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = rotation_matrix
        c2w[:3, 3] = position
        
        # Compute world-to-camera matrix (W2C, extrinsic)
        w2c = np.linalg.inv(c2w)
        w2c_4x3 = w2c[:3, :]  # Take first 3 rows (4x3 extrinsic format)
        
        # Update GUI text with formatted extrinsic matrix
        if self.gui_manual_extrinsic_text is not None:
            extrinsic_str = "**World to Manual Cam Extrinsic (4x3):**\n```\n"
            extrinsic_str += (
                f"[{w2c_4x3[0, 0]:8.4f}, {w2c_4x3[0, 1]:8.4f}, "
                f"{w2c_4x3[0, 2]:8.4f}, {w2c_4x3[0, 3]:8.4f}],\n"
            )
            extrinsic_str += (
                f"[{w2c_4x3[1, 0]:8.4f}, {w2c_4x3[1, 1]:8.4f}, "
                f"{w2c_4x3[1, 2]:8.4f}, {w2c_4x3[1, 3]:8.4f}],\n"
            )
            extrinsic_str += (
                f"[{w2c_4x3[2, 0]:8.4f}, {w2c_4x3[2, 1]:8.4f}, "
                f"{w2c_4x3[2, 2]:8.4f}, {w2c_4x3[2, 3]:8.4f}]\n"
            )
            extrinsic_str += "```"
            self.gui_manual_extrinsic_text.content = extrinsic_str


    def _rebuild_scene(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        spatial_subsample: int = self.gui_s_sub.value
        temporal_subsample: int = self.gui_t_sub.value
        
        # GlobalContext에서 설정값 가져오기
        use_mean_bg = self.global_context().use_mean_bg

        rays: np.ndarray | None = None
        first_frame_y: np.ndarray | None = None

        self.client.scene.reset()
        
        # ! 원래대로 하고 싶으면 밑의 3줄 주석 처리
        # 크로마키 그린 배경 재설정 (scene.reset()이 배경을 지우므로)
        # chroma_green_bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # chroma_green_bg[:, :] = [0, 177, 64]
        # self.client.scene.set_background_image(image=chroma_green_bg, format="png")
        
        self.client.camera.fov = np.deg2rad(self.gui_fov.value)
        self.scene_frame_handles = []
        
        # Add ego camera frustums if take_uuid is provided
        if (self.global_context().take_uuid is not None and 
            self.global_context().ego_extrinsics_list is not None and
            self.global_context().exo_cam_to_world is not None):
            self._add_ego_camera_frustums()
            self._add_ego_trajectory()  # Add trajectory visualization
            self._add_ego4d_world_axes()
        
        # Mean background depth 계산 (use_mean_bg가 true이고 아직 계산되지 않은 경우)
        if use_mean_bg and (self.global_context().mean_depth is None or self.global_context().mean_rgb is None):
            logger.info("Computing mean background depth for the first time...")
            mean_depth, mean_rgb = compute_mean_background_depth(
                current_artifact, spatial_subsample
            )
            # GlobalContext에 저장
            self.global_context().mean_depth = mean_depth
            self.global_context().mean_rgb = mean_rgb

        def none_it(inner_it):
            try:
                for item in inner_it:
                    yield item
            except FileNotFoundError:
                while True:
                    yield None, None

        # 마스크 데이터 로더 추가
        def none_it_mask(inner_it):
            try:
                for item in inner_it:
                    yield item
            except FileNotFoundError:
                while True:
                    yield None, None

        for frame_idx, (c2w, (_, rgb), intr, camera_type, (_, depth), (_, instance_mask)) in enumerate(
            zip(
                read_pose_artifacts(current_artifact.pose_path)[1].matrix().numpy(),
                read_rgb_artifacts(current_artifact.rgb_path),
                *read_intrinsics_artifacts(current_artifact.intrinsics_path, current_artifact.camera_type_path)[1:3],
                none_it(read_depth_artifacts(current_artifact.depth_path)),
                none_it_mask(read_instance_artifacts(current_artifact.mask_path)),
            )
        ):
            if frame_idx % temporal_subsample != 0:
                continue

            # Current frame size (needed for both default and GT paths)
            frame_height, frame_width = rgb.shape[:2]

            # Optionally override with exo GT intrinsics (scaled to current frame size)
            if self.global_context().use_exo_intrinsic_gt and self.global_context().gt_exo_intrinsics_K is not None and self.global_context().gt_exo_image_size is not None:
                src_w, src_h = self.global_context().gt_exo_image_size
                sx = frame_width / float(src_w)
                sy = frame_height / float(src_h)
                K = self.global_context().gt_exo_intrinsics_K.copy()
                K[0, 0] *= sx
                K[1, 1] *= sy
                K[0, 2] *= sx
                K[1, 2] *= sy
                intr = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
            pinhole_intr = camera_type.build_camera_model(intr).pinhole().intrinsics
            fov = 2 * np.arctan2(frame_height / 2, pinhole_intr[0].item())

            # RGB 처리 - use_mean_bg에 따라 다른 RGB 사용
            if not (use_mean_bg and self.global_context().mean_rgb is not None):
                # 일반적인 프레임별 RGB 사용
                sampled_rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
                sampled_rgb = sampled_rgb[::spatial_subsample, ::spatial_subsample]

            if first_frame_y is None:
                first_frame_y = c2w[:3, 1]
                self.client.scene.set_up_direction(-first_frame_y)

            if rays is None:
                camera_model = camera_type.build_camera_model(intr)
                disp_v, disp_u = torch.meshgrid(
                    torch.arange(frame_height).float()[::spatial_subsample],
                    torch.arange(frame_width).float()[::spatial_subsample],
                    indexing="ij",
                )
                if camera_type == CameraType.PANORAMA:
                    disp_v = disp_v / (frame_height - 1)
                    disp_u = disp_u / (frame_width - 1)
                disp = torch.ones_like(disp_v)
                pts, _, _ = camera_model.iproj_disp(disp, disp_u, disp_v)
                rays = pts[..., :3].numpy()
                if camera_type != CameraType.PANORAMA:
                    rays /= rays[..., 2:3]

            # Point cloud 처리 - use_mean_bg에 따라 다른 방식 사용
            dynamic_pcd = None
            dynamic_colors = None
            
            if use_mean_bg and self.global_context().mean_depth is not None:
                # Mean background depth 사용
                logger.info(f"Frame {frame_idx}: Using pre-computed mean background depth with nanmean")
                
                # GPU tensor를 numpy로 변환
                mean_depth_np = self.global_context().mean_depth.cpu().numpy()
                mean_rgb_np = self.global_context().mean_rgb.cpu().numpy()
                
                # Background point cloud 계산 (mean depth 사용)
                pcd = rays * mean_depth_np[..., None]
                
                # Mean depth에서 유효한 픽셀 마스크 (0이 아닌 값들)
                depth_mask = mean_depth_np > 0
                
                # RGB 색상도 mean RGB 사용 (0-255 범위로 변환)
                sampled_rgb = (mean_rgb_np * 255).astype(np.uint8)
                
                # Dynamic objects 추가 (현재 프레임의 실제 depth 사용)
                show_dynamic = True  # 기본값
                if hasattr(self, 'gui_show_dynamic_objects') and self.gui_show_dynamic_objects is not None:
                    show_dynamic = self.gui_show_dynamic_objects.value
                
                if depth is not None and instance_mask is not None and show_dynamic:
                    dynamic_pcd, dynamic_colors = build_dynamic_points_for_frame(
                        current_artifact, frame_idx, c2w, rgb, depth, instance_mask, 
                        camera_type, intr, spatial_subsample
                    )
                
                logger.info(f"Frame {frame_idx}: Using mean background with {np.sum(depth_mask)} background points + {len(dynamic_pcd) if dynamic_pcd is not None else 0} dynamic points")
                
            elif depth is not None:
                # 원래 코드 (프레임별 depth 사용)
                pcd = rays * depth.numpy()[::spatial_subsample, ::spatial_subsample, None]                
                depth_mask = reliable_depth_mask_range(depth)[::spatial_subsample, ::spatial_subsample].numpy()
                
                # 인스턴스 마스크를 이용한 동적 객체 필터링 (GUI 설정에 따라)
                if instance_mask is not None and self.gui_filter_dynamic_objects.value:
                    # 인스턴스 마스크를 numpy로 변환
                    instance_mask_np = instance_mask.cpu().numpy() if hasattr(instance_mask, 'cpu') else instance_mask
                    # 배경(0)만 유지하고 동적 객체(1=person, 2=car 등) 필터링
                    static_mask = (instance_mask_np == 0)
                    # 공간 서브샘플링 적용
                    static_mask_sub = static_mask[::spatial_subsample, ::spatial_subsample]
                    # 깊이 마스크와 정적 마스크 결합
                    depth_mask = depth_mask & static_mask_sub
                    logger.info(f"Frame {frame_idx}: Applied instance mask filtering (kept {np.sum(depth_mask)} / {depth_mask.size} points)")
            else:
                pcd, depth_mask = None, None

            frame_node = self._make_frame_nodes(
                frame_idx,
                c2w,
                sampled_rgb,
                fov,
                pcd,
                depth_mask,
                dynamic_pcd,
                dynamic_colors,
            )
            self.scene_frame_handles.append(frame_node)

    def _make_frame_nodes(
        self,
        frame_idx: int,
        c2w: np.ndarray,
        rgb: np.ndarray,
        fov: float,
        pcd: np.ndarray | None,
        pcd_mask: np.ndarray | None = None,
        dynamic_pcd: np.ndarray | None = None,
        dynamic_colors: np.ndarray | None = None,
    ) -> SceneFrameHandle:
        handle = self.client.scene.add_frame(
            f"/frames/t{frame_idx}",
            axes_length=0.05,
            axes_radius=0.005,
            wxyz=tf.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
        )
        frame_height, frame_width = rgb.shape[:2]

        frame_thumbnail = Image.fromarray(rgb)
        frame_thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
        frustum_handle = self.client.scene.add_camera_frustum(
            f"/frames/t{frame_idx}/frustum",
            fov=fov,
            aspect=frame_width / frame_height,
            scale=self.gui_frustum_size.value,
            image=np.array(frame_thumbnail),
        )

        # Background point cloud 처리
        if pcd is not None:
            pcd = pcd.reshape(-1, 3)
            rgb = rgb.reshape(-1, 3)
            if pcd_mask is not None:
                pcd_mask = pcd_mask.reshape(-1)
                pcd = pcd[pcd_mask]
                rgb = rgb[pcd_mask]
            pcd_handle = self.client.scene.add_point_cloud(
                name=f"/frames/t{frame_idx}/point_cloud_bg",
                points=pcd,
                colors=rgb,
                point_size=self.gui_point_size.value,
                point_shape="rounded",
            )
        else:
            pcd_handle = None

        # Dynamic point cloud 처리
        dynamic_pcd_handle = None
        if dynamic_pcd is not None and len(dynamic_pcd) > 0:
            # Dynamic objects를 카메라 좌표계에서 월드 좌표계로 변환
            dynamic_world = (c2w[:3, :3] @ dynamic_pcd.T + c2w[:3, 3:4]).T
            
            dynamic_pcd_handle = self.client.scene.add_point_cloud(
                name=f"/frames/t{frame_idx}/point_cloud_dynamic",
                points=dynamic_world,
                colors=dynamic_colors,
                point_size=self.gui_point_size.value * 1.2,  # Dynamic objects를 조금 더 크게
                point_shape="rounded",
            )
            logger.info(f"Frame {frame_idx}: Added {len(dynamic_world)} dynamic points to scene")

        return SceneFrameHandle(
            frame_handle=handle,
            frustum_handle=frustum_handle,
            pcd_handle=pcd_handle,
            dynamic_pcd_handle=dynamic_pcd_handle,
        )

    def _incr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value + 1) % len(self.scene_frame_handles)

    def _decr_timestep(self):
        if self.gui_timestep is not None:
            self.gui_timestep.value = (self.gui_timestep.value - 1) % len(self.scene_frame_handles)

    def _rebuild_playback_gui(self):
        current_artifact = self.global_context().artifacts[self.gui_id.value]
        self.gui_name.value = current_artifact.artifact_name
        if self.gui_playback_handle is not None:
            self.gui_playback_handle.remove()
        self.gui_playback_handle = self.client.gui.add_folder("Playback")

        with self.gui_playback_handle:
            self.gui_timestep = self.client.gui.add_slider(
                "Timeline", min=0, max=len(self.scene_frame_handles) - 1, step=1, initial_value=0
            )
            gui_frame_control = self.client.gui.add_button_group("Control", options=["Prev", "Next"])
            self.gui_framerate = self.client.gui.add_slider("FPS", min=0, max=30, step=1.0, initial_value=15)
            #### added ####
            self.gui_playing = self.client.gui.add_checkbox("Playing", initial_value=False)
            self.gui_show_all_frames = self.client.gui.add_checkbox("Show all frames", initial_value=False)

            @self.gui_show_all_frames.on_update
            async def _(_):
                # 토글 켜면 모든 프레임 보이기, 끄면 타임라인 위치만 보이기
                show_all = self.gui_show_all_frames.value
                with self.client.atomic():
                    if show_all:
                        for h in self.scene_frame_handles:
                            h.visible = True
                        # Show all ego frustums
                        if hasattr(self, 'ego_frustum_handles') and self.ego_frustum_handles:
                            for ego_frustum in self.ego_frustum_handles:
                                ego_frustum.visible = True
                    else:
                        for i, h in enumerate(self.scene_frame_handles):
                            h.visible = (i == self.gui_timestep.value)
                        # Show only current frame's ego frustum
                        if hasattr(self, 'ego_frustum_handles') and self.ego_frustum_handles:
                            for i, ego_frustum in enumerate(self.ego_frustum_handles):
                                ego_frustum.visible = (i == self.gui_timestep.value)
            ###############

            @gui_frame_control.on_click
            async def _(_) -> None:
                if gui_frame_control.value == "Prev":
                    self._decr_timestep()
                else:
                    self._incr_timestep()

            self.current_displayed_timestep = self.gui_timestep.value

            @self.gui_timestep.on_update
            async def _(_) -> None:
                current_timestep = self.gui_timestep.value
                prev_timestep = self.current_displayed_timestep
                with self.client.atomic():
                    if not (self.gui_show_all_frames and self.gui_show_all_frames.value): # added
                        self.scene_frame_handles[current_timestep].visible = True
                        self.scene_frame_handles[prev_timestep].visible = False
                        
                        # Update ego camera frustum visibility
                        if hasattr(self, 'ego_frustum_handles') and self.ego_frustum_handles:
                            # Hide all ego frustums first
                            for ego_frustum in self.ego_frustum_handles:
                                ego_frustum.visible = False
                            # Show only current frame's ego frustum
                            if current_timestep < len(self.ego_frustum_handles):
                                self.ego_frustum_handles[current_timestep].visible = True
                self.current_displayed_timestep = current_timestep

    def cleanup(self):
        logger.info(f"Client {self.client.client_id} disconnected")

    @classmethod
    def global_context(cls) -> GlobalContext:
        global _global_context
        assert _global_context is not None, "Global context not initialized"
        return _global_context


def load_ego_camera_poses(camera_pose_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Load ego camera poses from JSON file.
    
    Returns:
        intrinsics: 3x3 camera intrinsics matrix
        extrinsics_list: List of 3x4 camera extrinsics matrices for each frame (world-to-camera)
    """
    with open(camera_pose_path, 'r') as f:
        data = json.load(f)
    
    # Find aria camera (ego view)
    aria_key = None
    for key in data.keys():
        if key.startswith('aria'):
            aria_key = key
            break
    
    if aria_key is None:
        raise ValueError("No aria camera found in camera pose file")
    
    logger.info(f"Using ego camera: {aria_key}")
    
    # Extract camera intrinsics (3x3)
    intrinsics = np.array(data[aria_key]['camera_intrinsics'])
    
    # Extract camera extrinsics for each frame
    # camera_extrinsics is a dictionary with string keys "0", "1", etc.
    extrinsics_dict = data[aria_key]['camera_extrinsics']
    extrinsics_list = []
    
    # Get all keys and sort them numerically
    frame_indices = sorted([int(k) for k in extrinsics_dict.keys()])
    
    for frame_idx in frame_indices:
        # Load as 3x4 matrix from JSON
        extrinsics = np.array(extrinsics_dict[str(frame_idx)])
        extrinsics_list.append(extrinsics)
    
    logger.info(f"Loaded intrinsics: {intrinsics.shape}")
    logger.info(f"Loaded {len(extrinsics_list)} extrinsics matrices (W2C - world-to-camera format)")
    
    return intrinsics, extrinsics_list


def load_exo_camera_pose(exo_camera_pose_path: str, cam_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load exo camera pose from Ego4D trajectory CSV file.
    
    Args:
        exo_camera_pose_path: Path to gopro_calibs.csv file
        cam_id: Camera ID to extract (required, e.g., 'cam01', 'cam02', 'gp01', 'gp02', etc.)
    
    Returns:
        T_world_to_cam: 4x4 transformation matrix from world to camera coordinates
        T_cam_to_world: 4x4 transformation matrix from camera to world coordinates
        camera_center: 3D camera center position in world coordinates
    """
    # Read CSV file
    df = pd.read_csv(exo_camera_pose_path)
    
    # Find the camera row
    cam_row = df[df['cam_uid'] == cam_id]
    if cam_row.empty:
        raise ValueError(f"Camera {cam_id} not found in {exo_camera_pose_path}")
    
    cam_row = cam_row.iloc[0]
    
    # Extract translation and quaternion
    # Ego-Exo4D CSV format: tx_world_cam, qx_world_cam etc. represent T_cam_to_world (T_wc)
    tx = cam_row['tx_world_cam']
    ty = cam_row['ty_world_cam'] 
    tz = cam_row['tz_world_cam']
    camera_center = np.array([tx, ty, tz])  # Camera center in world coordinates
    
    # Extract quaternion (qx, qy, qz, qw) - rotation from camera to world (R_wc)
    qx = cam_row['qx_world_cam']
    qy = cam_row['qy_world_cam']
    qz = cam_row['qz_world_cam']
    qw = cam_row['qw_world_cam']
    quaternion = np.array([qx, qy, qz, qw])
    
    # Convert quaternion to rotation matrix (camera to world)
    R_cam_to_world = R.from_quat(quaternion).as_matrix()
    
    # Create camera-to-world transformation (T_wc)
    T_cam_to_world = np.eye(4)
    T_cam_to_world[:3, :3] = R_cam_to_world  # Camera to world rotation
    T_cam_to_world[:3, 3] = camera_center    # Camera center position in world
    
    # Create world-to-camera transformation (T_cw = inv(T_wc))
    T_world_to_cam = np.eye(4)
    T_world_to_cam[:3, :3] = R_cam_to_world.T  # Inverse rotation (world to camera)
    T_world_to_cam[:3, 3] = -R_cam_to_world.T @ camera_center  # Proper translation
    
    logger.info(f"Loaded exo camera {cam_id} pose from {exo_camera_pose_path}")
    logger.info(f"Camera center in world: {camera_center}")
    logger.info(f"Quaternion (cam->world): {quaternion}")
    
    return T_world_to_cam, T_cam_to_world, camera_center


def load_semidense_points(take_name: str, 
                         inverse_distance_std_threshold: float = 0.001,
                         distance_std_threshold: float = 0.15) -> Optional[np.ndarray]:
    """
    Load Ego4D semi-dense point cloud using projectaria-tools.
    
    Args:
        take_name: Take name (e.g., 'georgiatech_cooking_01_03_2')
        inverse_distance_std_threshold: Threshold for inverse distance std filtering
        distance_std_threshold: Threshold for distance std filtering
    
    Returns:
        points_world: Nx3 array of 3D points in world coordinates, or None if failed
    """
    if not PROJECTARIA_AVAILABLE:
        logger.warning("projectaria-tools not available, skipping semi-dense point cloud loading")
        return None
    
    # Construct path to semi-dense points
    semidense_path = f"/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/takes/{take_name}/trajectory/semidense_points.csv.gz"
    
    try:
        logger.info(f"Loading semi-dense points from: {semidense_path}")
        
        # Load global point cloud using projectaria-tools
        points = mps.read_global_point_cloud(semidense_path)
        logger.info(f"Loaded {len(points)} raw semi-dense points")
        
        # Filter points based on confidence thresholds
        filtered_points = filter_points_from_confidence(
            points, 
            inverse_distance_std_threshold, 
            distance_std_threshold
        )
        logger.info(f"Filtered to {len(filtered_points)} confident semi-dense points")
        
        # Extract 3D positions in world coordinates
        points_world = np.array([point.position_world for point in filtered_points])
        logger.info(f"Extracted {points_world.shape[0]} 3D points for visualization")
        
        return points_world
        
    except Exception as e:
        logger.error(f"Failed to load semi-dense points: {e}")
        return None


def load_exo_gt_intrinsics_from_camera_pose(take_uuid: str, cam_id: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """
    Load exo camera pinhole intrinsics from camera_pose/{take_uuid}.json.
    Returns (K 3x3, (width, height)) or (None, None) if not found.
    """
    pose_path = f"/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/annotations/ego_pose/train/camera_pose/{take_uuid}.json"
    try:
        with open(pose_path, 'r') as f:
            data = json.load(f)
        
        # Check if cam_id exists in the JSON
        if cam_id not in data:
            logger.warning(f"Camera {cam_id} not found in camera_pose JSON")
            return None, None
        
        cam_data = data[cam_id]
        if 'camera_intrinsics' not in cam_data:
            logger.warning(f"No camera_intrinsics found for {cam_id}")
            return None, None
        
        K = cam_data['camera_intrinsics']
        
        # Convert to numpy array
        if isinstance(K, list) and len(K) == 3 and all(isinstance(row, list) and len(row) == 3 for row in K):
            K_mat = np.array(K, dtype=float)
        else:
            logger.warning(f"Invalid camera_intrinsics format for {cam_id}: {K}")
            return None, None
        
        # For camera_pose JSON, we need to infer the image size from the intrinsics
        # The cx, cy values typically correspond to the image center
        cx, cy = K_mat[0, 2], K_mat[1, 2]
        # Assume standard GoPro resolution (3840x2160) based on cx=1920, cy=1080
        width, height = int(cx * 2), int(cy * 2)
        
        logger.info(f"Loaded GT intrinsics for {cam_id}: fx={K_mat[0,0]:.2f}, fy={K_mat[1,1]:.2f}, cx={cx:.1f}, cy={cy:.1f}, size={width}x{height}")
        return K_mat, (width, height)
        
    except FileNotFoundError:
        logger.warning(f"camera_pose/{take_uuid}.json not found")
    except Exception as e:
        logger.warning(f"Failed to parse camera_pose JSON: {e}")
    return None, None


def parse_camera_id_from_input_dir(input_dir: str) -> str:
    """
    Parse camera ID from input directory path.
    
    Args:
        input_dir: Input directory path like 'vipe_results/take_name/cam04_static_vda_fixedcam_slammap' or 'vipe_results/take_name/gp02_static_vda_fixedcam_slammap'
    
    Returns:
        Camera ID string like 'cam04' or 'gp02'
    """
    import re
    
    # Extract camera ID pattern (cam + 2 digits or gp + 2 digits)
    pattern = r'(cam\d{2}|gp\d{2})'
    match = re.search(pattern, input_dir)
    
    if match:
        cam_id = match.group(0)
        logger.info(f"Parsed camera ID '{cam_id}' from input_dir: {input_dir}")
        return cam_id
    else:
        # No fallback - raise error if camera ID cannot be parsed
        raise ValueError(f"Could not parse camera ID from input_dir: {input_dir}. Expected format with 'camXX' or 'gpXX' pattern (e.g., cam01, cam02, gp01, gp02, etc.)")


def get_host_ip() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            # Doesn't even have to be reachable
            s.connect(("8.8.8.8", 1))
            internal_ip = s.getsockname()[0]
        except Exception:
            internal_ip = "127.0.0.1"
    return internal_ip


def run_viser(base_path: Path, port: int = 20540, use_mean_bg: bool = False, take_uuid: str = None, start_frame: int = None, use_exo_intrinsic_gt: bool = False, ego_manual: bool = False):
    # Get list of artifacts.
    logger.info(f"Loading artifacts from {base_path}")
    artifacts: list[ArtifactPath] = list(ArtifactPath.glob_artifacts(base_path, use_video=True))
    if len(artifacts) == 0:
        logger.error("No artifacts found. Exiting.")
        return

    # Load ego camera data if take_uuid is provided
    ego_extrinsics_list = None
    exo_cam_to_world = None
    
    if take_uuid is not None and start_frame is not None:
        logger.info(f"Loading ego camera data for take_uuid: {take_uuid}, start_frame: {start_frame}")
        
        try:
            # Parse camera ID and robustly derive take_name from data path
            first_artifact = artifacts[0]
            cam_id = parse_camera_id_from_input_dir(str(first_artifact.base_path))
            try:
                bp = base_path if isinstance(base_path, Path) else Path(str(base_path))
                parts = bp.parts
                take_name = None
                # Case 1: find after 'vipe_results'
                if 'vipe_results' in parts:
                    idx = parts.index('vipe_results')
                    if len(parts) > idx + 1:
                        take_name = parts[idx + 1]
                # Case 2: user passed camera folder directly like .../{take}/cam02_...
                if take_name is None:
                    import re
                    if re.match(r'^(cam\d{2}|gp\d{2})', bp.name):
                        take_name = bp.parent.name
                # Fallback: use artifact parent
                if take_name is None:
                    take_name = Path(first_artifact.base_path).parent.name
            except Exception:
                take_name = Path(first_artifact.base_path).parent.name
            logger.info(f"Derived take_name='{take_name}' from data_path='{base_path}'")
            
            # Construct paths
            ego_pose_path = f"/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/annotations/ego_pose/train/camera_pose/{take_uuid}.json"
            # Exo trajectory path requires take_name (not ego UUID)
            exo_pose_path = f"/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/takes/{take_name}/trajectory/gopro_calibs.csv"
            
            # Load ego camera poses (only extrinsics needed for frustum visualization)
            _, ego_extrinsics_full = load_ego_camera_poses(ego_pose_path)
            
            # Calculate actual number of frames from ViPE artifacts
            try:
                # Use the same method as in render_vipe_pointcloud.py
                pose_inds, _ = read_pose_artifacts(first_artifact.pose_path)
                num_frames = len(pose_inds)
                logger.info(f"Detected {num_frames} frames from ViPE artifacts")
            except Exception as e:
                logger.warning(f"Failed to read pose artifacts: {e}")
                # Fallback: count RGB frames
                try:
                    rgb_frames = list(read_rgb_artifacts(first_artifact.rgb_path))
                    num_frames = len(rgb_frames)
                    logger.info(f"Fallback: detected {num_frames} frames from RGB artifacts")
                except Exception as e2:
                    logger.warning(f"Failed to read RGB artifacts: {e2}")
                    num_frames = 50  # Final fallback
                    logger.warning(f"Using fallback frame count: {num_frames}")
            
            # Slice ego extrinsics based on start_frame and number of frames in artifacts
            ego_extrinsics_list = ego_extrinsics_full[start_frame:start_frame + num_frames]
            
            # Load exo camera pose
            _, exo_cam_to_world, _ = load_exo_camera_pose(exo_pose_path, cam_id)
            
            logger.info(f"Successfully loaded ego camera data: {len(ego_extrinsics_list)} frames")
            logger.info(f"Successfully loaded exo camera pose for {cam_id}")
            
        except Exception as e:
            logger.warning(f"Failed to load ego camera data: {e}")
            logger.warning("Continuing without ego camera visualization")
            take_uuid = None
            start_frame = None

    # Load semi-dense point cloud if take_uuid is provided
    semidense_points = None
    if take_uuid is not None:
        try:
            # Extract take_name from base_path
            bp = base_path if isinstance(base_path, Path) else Path(str(base_path))
            parts = bp.parts
            take_name = None
            # Case 1: find after 'vipe_results'
            if 'vipe_results' in parts:
                idx = parts.index('vipe_results')
                if len(parts) > idx + 1:
                    take_name = parts[idx + 1]
            # Case 2: user passed camera folder directly like .../{take}/cam02_...
            if take_name is None:
                import re
                for part in reversed(parts):
                    if re.match(r'^[a-zA-Z0-9_]+_cooking_\d+_\d+_\d+$', part):
                        take_name = part
                        break
            
            if take_name:
                logger.info(f"Loading semi-dense point cloud for take: {take_name}")
                semidense_points = load_semidense_points(take_name)
                if semidense_points is not None:
                    logger.info(f"Successfully loaded {semidense_points.shape[0]} semi-dense points")
                else:
                    logger.warning("Failed to load semi-dense points")
            else:
                logger.warning("Could not extract take_name for semi-dense point loading")
                
        except Exception as e:
            logger.warning(f"Failed to load semi-dense points: {e}")

    # Load exo GT intrinsics if requested
    gt_K = None
    gt_size = None
    if use_exo_intrinsic_gt:
        try:
            # Derive take_name and cam_id as above
            first_artifact = artifacts[0]
            cam_id = parse_camera_id_from_input_dir(str(first_artifact.base_path))
            bp = base_path if isinstance(base_path, Path) else Path(str(base_path))
            parts = bp.parts
            take_name = None
            if 'vipe_results' in parts:
                idx = parts.index('vipe_results')
                if len(parts) > idx + 1:
                    take_name = parts[idx + 1]
            if take_name is None:
                take_name = Path(first_artifact.base_path).parent.name
            gt_K, gt_size = load_exo_gt_intrinsics_from_camera_pose(take_uuid, cam_id)
            if gt_K is not None:
                logger.info(f"Loaded exo GT intrinsics from camera_pose.json with size {gt_size}")
            else:
                logger.warning("Failed to find exo GT intrinsics in camera_pose.json; will use ViPE intrinsics")
        except Exception as e:
            logger.warning(f"Failed to load exo GT intrinsics: {e}")

    global _global_context
    _global_context = GlobalContext(
        artifacts=sorted(artifacts, key=lambda x: x.artifact_name),
        use_mean_bg=use_mean_bg,
        use_exo_intrinsic_gt=use_exo_intrinsic_gt,
        take_uuid=take_uuid,
        start_frame=start_frame,
        ego_extrinsics_list=ego_extrinsics_list,
        exo_cam_to_world=exo_cam_to_world,
        semidense_points=semidense_points,
        gt_exo_intrinsics_K=gt_K,
        gt_exo_image_size=gt_size,
        ego_manual=ego_manual
    )

    # 새 코드: 모든 인터페이스에서 수신 대기
    server = viser.ViserServer(host="0.0.0.0", port=port, verbose=False)
    # 원래 코드 (주석 처리)
    # server = viser.ViserServer(host=get_host_ip(), port=port, verbose=False)
    client_closures: dict[int, ClientClosures] = {}

    @server.on_client_connect
    async def _(client: viser.ClientHandle):
        client_closures[client.client_id] = ClientClosures(client)

    @server.on_client_disconnect
    async def _(client: viser.ClientHandle):
        # wait synchronously in this function for task to be finished.
        await client_closures[client.client_id].stop()
        del client_closures[client.client_id]

    while True:
        try:
            time.sleep(10.0)
        except KeyboardInterrupt:
            logger.info("Ctrl+C detected. Shutting down server...")
            break
    server.stop()


def main():
    parser = argparse.ArgumentParser(description="3D Visualizer")
    parser.add_argument("base_path", type=Path, help="Base path for the visualizer")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=20540,
        help="Port number for the viser server.",
    )
    parser.add_argument(
        "--use_mean_bg", 
        action="store_true", 
        help="Use nanmean background instead of standard background"
    )
    parser.add_argument(
        "--take_uuid",
        type=str,
        help="Take UUID for ego camera pose visualization (optional)"
    )
    parser.add_argument(
        "--start_frame",
        type=int,
        help="Start frame for ego camera pose visualization (required if --take_uuid is provided)"
    )
    parser.add_argument(
        "--ego_manual",
        action="store_true",
        help="Enable manual ego camera control with transform handles"
    )
    args = parser.parse_args()

    # Validate that start_frame is provided if take_uuid is provided
    if args.take_uuid is not None and args.start_frame is None:
        parser.error("--start_frame is required when --take_uuid is provided")

    run_viser(args.base_path, args.port, args.use_mean_bg, args.take_uuid, args.start_frame, False, args.ego_manual)


if __name__ == "__main__":
    main()
