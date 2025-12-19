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


import logging

from typing import Iterator

import numpy as np
import torch

from vipe.priors.depth import DepthEstimationInput, make_depth_model
from vipe.priors.depth.alignment import align_inv_depth_to_depth, align_metric_depth_to_depth
from vipe.priors.depth.priorda import PriorDAModel
from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
from vipe.priors.geocalib import GeoCalib
from vipe.priors.track_anything import TrackAnythingPipeline
from vipe.slam.interface import SLAMOutput
from vipe.streams.base import CachedVideoStream, FrameAttribute, StreamProcessor, VideoFrame, VideoStream
from vipe.utils.cameras import CameraType
from vipe.utils.logging import pbar
from vipe.utils.misc import unpack_optional
from vipe.utils.morph import erode


logger = logging.getLogger(__name__)


class IntrinsicEstimationProcessor(StreamProcessor):
    """Override existing intrinsics with estimated intrinsics."""

    def __init__(self, video_stream: VideoStream, gap_sec: float = 1.0) -> None:
        super().__init__()
        gap_frame = int(gap_sec * video_stream.fps())
        gap_frame = min(gap_frame, (len(video_stream) - 1) // 2)
        self.sample_frame_inds = [0, gap_frame, gap_frame * 2]
        self.fov_y = -1.0
        self.camera_type = CameraType.PINHOLE
        self.distortion: list[float] = []

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        assert self.fov_y > 0, "FOV not set"
        frame_height, frame_width = frame.size()
        fx = fy = frame_height / (2 * np.tan(self.fov_y / 2))
        frame.intrinsics = torch.as_tensor(
            [fx, fy, frame_width / 2, frame_height / 2] + self.distortion,
        ).float()
        frame.camera_type = self.camera_type
        return frame


class GeoCalibIntrinsicsProcessor(IntrinsicEstimationProcessor):
    def __init__(
        self,
        video_stream: VideoStream,
        gap_sec: float = 1.0,
        camera_type: CameraType = CameraType.PINHOLE,
    ) -> None:
        super().__init__(video_stream, gap_sec)

        is_pinhole = camera_type == CameraType.PINHOLE
        weights = "pinhole" if is_pinhole else "distorted"

        model = GeoCalib(weights=weights).cuda()
        indexable_stream = CachedVideoStream(video_stream)

        if is_pinhole:
            sample_frames = torch.stack([indexable_stream[i].rgb.moveaxis(-1, 0) for i in self.sample_frame_inds])
            res = model.calibrate(
                sample_frames,
                shared_intrinsics=True,
            )
        else:
            # Use first frame for calibration
            camera_model = {
                CameraType.PINHOLE: "pinhole",
                CameraType.MEI: "simple_mei",
            }[camera_type]
            res = model.calibrate(
                indexable_stream[self.sample_frame_inds[0]].rgb.moveaxis(-1, 0)[None],
                camera_model=camera_model,
            )

        self.fov_y = res["camera"].vfov[0].item()
        self.camera_type = camera_type

        if not is_pinhole:
            # Assign distortion parameter
            self.distortion = [res["camera"].dist[0, 0].item()]


class GTIntrinsicsProcessor(StreamProcessor):
    """Load ground truth intrinsics from camera_pose JSON files."""
    
    def __init__(
        self,
        video_stream: VideoStream,
        take_uuid: str,
        camera_type: CameraType = CameraType.PINHOLE,
        take_name: str | None = None,
        start_frame: int | None = None,
    ) -> None:
        super().__init__()
        self.camera_type = camera_type
        self.take_uuid = take_uuid
        self.camera_name = video_stream.name()  # e.g., "cam01" or "exo_GT"
        
        from pathlib import Path
        import json
        
        # Check if camera_name is "exo_GT" - need take_name and start_frame
        if self.camera_name == "exo_GT":
            # take_name and start_frame should be provided by caller
            if take_name is None:
                raise ValueError("take_name must be provided for exo_GT camera")
            if start_frame is None:
                raise ValueError("start_frame must be provided for exo_GT camera")
            
            # Load camera params JSON
            params_json_path = Path("/home/nas_main/kinamkim/Repos/Ego-Renderer-from-ViPE/EGO4D_4DNEX_DATASET/data_wan/ego_prior_datasets_split_with_camera_params.json")
            
            if not params_json_path.exists():
                raise FileNotFoundError(f"Camera params file not found: {params_json_path}")
            
            with open(params_json_path, 'r') as f:
                params_data = json.load(f)
            
            # Find matching entry in datasets list
            matching_entry = None
            for entry in params_data['datasets']:
                # Check if 'path' contains take_name and start_frame matches
                if take_name in entry['path'] and entry['source_frame_start'] == start_frame:
                    matching_entry = entry
                    break
            
            if matching_entry is None:
                raise ValueError(f"No matching entry found for take_name='{take_name}', start_frame={start_frame} in {params_json_path}")
            
            # Extract intrinsics from matching entry
            K_matrix = np.array(matching_entry["camera_intrinsics"])
            logging.info(f"Loaded GT intrinsics from ego_prior_datasets for exo_GT: take_name={take_name}, start_frame={matching_entry['source_frame_start']}")
        else:
            # Original logic for specific camera names (cam01, cam02, etc.)
            json_path = Path(f"/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/annotations/ego_pose/train/camera_pose/{take_uuid}.json")
            
            if not json_path.exists():
                raise FileNotFoundError(f"GT intrinsics file not found: {json_path}")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if self.camera_name not in data:
                raise KeyError(f"Camera '{self.camera_name}' not found in GT data. Available cameras: {list(data.keys())}")
            
            camera_data = data[self.camera_name]
            
            # Extract intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            K_matrix = np.array(camera_data["camera_intrinsics"])
        
        # Common intrinsics extraction
        self.gt_fx = K_matrix[0, 0]
        self.gt_fy = K_matrix[1, 1]
        self.gt_cy = K_matrix[1, 2]
        
        # Extract distortion coefficients (GeoCalibProcessor와 동일하게 처리)
        # PINHOLE이면 distortion 사용 안 함, 아니면 사용
        is_pinhole = camera_type == CameraType.PINHOLE
        if not is_pinhole:
            # For exo_GT from params JSON, distortion info might not be available
            distortion_coeffs = camera_data.get("distortion_coeffs", []) if self.camera_name != "exo_GT" else []
            # GeoCalibProcessor는 단일 distortion 값만 사용하므로 첫 번째 값만 사용
            if distortion_coeffs:
                self.distortion = [distortion_coeffs[0]]
            else:
                self.distortion = []
        else:
            self.distortion = []
        
        logging.info(f"Loaded GT intrinsics for {self.camera_name}: gt_fx={self.gt_fx:.2f}, gt_fy={self.gt_fy:.2f}, gt_cy={self.gt_cy:.2f}, camera_type={self.camera_type}, distortion={'present' if self.distortion else 'none'}")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INTRINSICS}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        """Apply GT intrinsics to frame, scaling to current frame size."""
        frame_height, frame_width = frame.size()
        
        # cx, cy는 현재 프레임의 실제 해상도 기준으로 사용
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        
        # fx, fy는 GT의 cy와 현재 프레임의 cy를 비교해서 height 기준으로 스케일링
        scale = cy / self.gt_cy
        fx_scaled = self.gt_fx * scale
        fy_scaled = self.gt_fy * scale
        
        # GeoCalibProcessor와 동일하게 처리: [fx, fy, cx, cy] + distortion
        frame.intrinsics = torch.as_tensor(
            [fx_scaled, fy_scaled, cx, cy] + self.distortion,
        ).float()
        frame.camera_type = self.camera_type
        return frame


class TrackAnythingProcessor(StreamProcessor):
    """
    A processor that tracks a mask caption in the video.
    """

    def __init__(
        self,
        mask_phrases: list[str],
        add_sky: bool,
        sam_run_gap: int = 30,
        mask_expand: int = 5,
    ) -> None:
        self.mask_phrases = mask_phrases
        self.sam_run_gap = sam_run_gap
        self.add_sky = add_sky

        if self.add_sky:
            self.mask_phrases.append(VideoFrame.SKY_PROMPT)

        self.tracker = TrackAnythingPipeline(self.mask_phrases, sam_points_per_side=50, sam_run_gap=self.sam_run_gap)
        self.mask_expand = mask_expand

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.INSTANCE, FrameAttribute.MASK}

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        frame.instance, frame.instance_phrases = self.tracker.track(frame)
        self.last_track_frame = frame.raw_frame_idx

        frame_instance_mask = frame.instance == 0
        if self.add_sky:
            # We won't mask out the sky.
            frame_instance_mask |= frame.sky_mask

        frame.mask = erode(frame_instance_mask, self.mask_expand)
        return frame


class AdaptiveDepthProcessor(StreamProcessor):
    """
    Compute projection of the SLAM map onto the current frames.
    If it's well-distributed, then use the fast map-prompted video depth model.
    If not, then use the slow metric depth + video depth alignment model.
    """

    def __init__(
        self,
        slam_output: SLAMOutput,
        view_idx: int = 0,
        model: str = "adaptive_unidepth-l_svda",
        share_depth_model: bool = False,
    ):
        super().__init__()
        self.slam_output = slam_output
        self.infill_target_pose = self.slam_output.get_view_trajectory(view_idx)
        assert view_idx == 0, "Adaptive depth processor only supports view_idx=0"
        assert not share_depth_model, "Adaptive depth processor does not support shared depth model"
        self.require_cache = True
        self.model = model

        try:
            prefix, metric_model, video_model = model.split("_")
            assert video_model in ["svda", "vda", "metric-vda"]
            if video_model == "metric-vda":
                self.video_depth_model = VideoDepthAnythingDepthModel(model="mvitl")
                self.is_metric_video = True
            else:
                self.video_depth_model = VideoDepthAnythingDepthModel(model="vits" if video_model == "svda" else "vitl")
                self.is_metric_video = False

        except ValueError:
            prefix, metric_model = model.split("_")
            video_model = None
            self.video_depth_model = None
            self.is_metric_video = False

        assert prefix == "adaptive", "Model name should start with 'adaptive_'"

        self.depth_model = make_depth_model(metric_model)
        self.prompt_model = PriorDAModel()
        self.update_momentum = 0.99

    def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
        raise NotImplementedError("AdaptiveDepthProcessor should not be called directly.")

    def update_attributes(self, previous_attributes: set[FrameAttribute]) -> set[FrameAttribute]:
        return previous_attributes | {FrameAttribute.METRIC_DEPTH}

    def _compute_uv_score(self, depth: torch.Tensor, patch_count: int = 10) -> float:
        h_shape = depth.size(0) // patch_count
        w_shape = depth.size(1) // patch_count
        depth_crop = (depth > 0)[: h_shape * patch_count, : w_shape * patch_count]
        depth_crop = depth_crop.reshape(patch_count, h_shape, patch_count, w_shape)
        depth_exist = depth_crop.any(dim=(1, 3))
        return depth_exist.float().mean().item()

    def _compute_video_da(self, frame_iterator: Iterator[VideoFrame]) -> tuple[torch.Tensor, list[VideoFrame]]:
        frame_list: list[np.ndarray] = []
        frame_data_list: list[VideoFrame] = []
        for frame in frame_iterator:
            frame_data_list.append(frame.cpu())
            frame_list.append(frame.rgb.cpu().numpy())

        estimation_result = self.video_depth_model.estimate(DepthEstimationInput(video_frame_list=frame_list))
        if self.is_metric_video:
            video_depth_result: torch.Tensor = unpack_optional(estimation_result.metric_depth)
        else:
            video_depth_result: torch.Tensor = unpack_optional(estimation_result.relative_inv_depth)
        return video_depth_result, frame_data_list

    def update_iterator(self, previous_iterator: Iterator[VideoFrame]) -> Iterator[VideoFrame]:
        # Determine the percentage score of the SLAM map.

        self.cache_scale_bias = None
        min_uv_score: float = 1.0

        if self.video_depth_model is not None:
            video_depth_result, data_iterator = self._compute_video_da(previous_iterator)
        else:
            video_depth_result = None
            data_iterator = previous_iterator

        for frame_idx, frame in pbar(enumerate(data_iterator), desc="Aligning depth"):
            # Convert back to GPU if not already.
            frame = frame.cuda()

            # Compute the minimum UV score only once at the 0-th frame.
            if frame_idx == 0:
                for test_frame_idx in range(self.slam_output.trajectory.shape[0]):
                    if test_frame_idx % 10 != 0:
                        continue
                    depth_infilled = self.slam_output.slam_map.project_map(
                        test_frame_idx,
                        0,
                        frame.size(),
                        unpack_optional(frame.intrinsics),
                        self.infill_target_pose[test_frame_idx],
                        unpack_optional(frame.camera_type),
                        infill=False,
                    )
                    uv_score = self._compute_uv_score(depth_infilled)
                    if uv_score < min_uv_score:
                        min_uv_score = uv_score

                logger.info(f"Minimum UV score: {min_uv_score:.4f}")

                # Decide once whether SLAM map is good enough and log the decision.
                self._use_slam_prompt = min_uv_score >= 0.3
                if self._use_slam_prompt:
                    logger.info(f"SLAM map will be used as prompt (min_uv_score={min_uv_score:.4f}).")
                else:
                    logger.info(f"SLAM map NOT used as prompt; falling back to metric model (min_uv_score={min_uv_score:.4f}).")

            # Use the previously decided flag for clarity and stable logging
            if not getattr(self, "_use_slam_prompt", False):
                focal_length = frame.intrinsics[0].item()
                prompt_result = self.depth_model.estimate(
                    DepthEstimationInput(rgb=frame.rgb.float().cuda(), focal_length=focal_length)
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(Metric)"
                logger.debug(f"Frame {frame_idx}: using metric depth prompt (uv={min_uv_score:.4f}).")
            else:
                depth_map = self.slam_output.slam_map.project_map(
                    frame_idx,
                    0,
                    frame.size(),
                    unpack_optional(frame.intrinsics),
                    self.infill_target_pose[frame_idx],
                    unpack_optional(frame.camera_type),
                    infill=False,
                )
                if frame.mask is not None:
                    depth_map = depth_map * frame.mask.float()
                prompt_result = self.prompt_model.estimate(
                    DepthEstimationInput(
                        rgb=frame.rgb.float().cuda(),
                        prompt_metric_depth=depth_map,
                    )
                ).metric_depth
                frame.information = f"uv={min_uv_score:.2f}(SLAM)"
                logger.debug(f"Frame {frame_idx}: using SLAM-prompted PriorDA (uv={min_uv_score:.4f}).")

            if video_depth_result is not None:
                if self.is_metric_video:
                    # For metric video depth models, align metric depth to depth
                    video_depth = video_depth_result[frame_idx]
                    
                    align_mask = video_depth > 1e-3
                    if frame.mask is not None:
                        align_mask = align_mask & frame.mask & (~frame.sky_mask)

                    try:
                        aligned_depth, scale, bias = align_metric_depth_to_depth(
                            video_depth,
                            prompt_result,
                            align_mask,
                        )
                        
                        # momentum update for metric video models
                        if self.cache_scale_bias is None:
                            self.cache_scale_bias = (scale, bias)
                        scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
                        bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
                        self.cache_scale_bias = (scale, bias)
                        
                        frame.metric_depth = aligned_depth
                    except RuntimeError:
                        # Fallback to video depth if alignment fails
                        frame.metric_depth = video_depth
                else:
                    video_depth_inv_depth = video_depth_result[frame_idx]

                    align_mask = video_depth_inv_depth > 1e-3
                    if frame.mask is not None:
                        align_mask = align_mask & frame.mask & (~frame.sky_mask)

                    try:
                        _, scale, bias = align_inv_depth_to_depth(
                            unpack_optional(video_depth_inv_depth),
                            prompt_result,
                            align_mask,
                        )
                    except RuntimeError:
                        scale, bias = self.cache_scale_bias

                    # momentum update
                    if self.cache_scale_bias is None:
                        self.cache_scale_bias = (scale, bias)
                    scale = self.cache_scale_bias[0] * self.update_momentum + scale * (1 - self.update_momentum)
                    bias = self.cache_scale_bias[1] * self.update_momentum + bias * (1 - self.update_momentum)
                    self.cache_scale_bias = (scale, bias)

                    video_inv_depth = video_depth_inv_depth * scale + bias
                    video_inv_depth[video_inv_depth < 1e-3] = 1e-3
                    frame.metric_depth = video_inv_depth.reciprocal()

            else:
                frame.metric_depth = prompt_result

            yield frame
