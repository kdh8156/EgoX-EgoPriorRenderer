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
import pickle

from pathlib import Path

import torch

from omegaconf import DictConfig

from vipe.slam.system import SLAMOutput, SLAMSystem
from vipe.streams.base import (
    AssignAttributesProcessor,
    FrameAttribute,
    MultiviewVideoList,
    ProcessedVideoStream,
    StreamProcessor,
    VideoStream,
)
from vipe.utils import io
from vipe.utils.cameras import CameraType
from vipe.utils.visualization import save_projection_video

from . import AnnotationPipelineOutput, Pipeline
from .processors import AdaptiveDepthProcessor, GeoCalibIntrinsicsProcessor, GTIntrinsicsProcessor, TrackAnythingProcessor


logger = logging.getLogger(__name__)


class DefaultAnnotationPipeline(Pipeline):
    def __init__(self, init: DictConfig, slam: DictConfig, post: DictConfig, output: DictConfig, assume_fixed_camera_pose: bool = False, use_exo_intrinsic_gt: str = None) -> None:
        super().__init__()
        self.init_cfg = init
        self.slam_cfg = slam
        self.post_cfg = post
        self.out_cfg = output
        self.assume_fixed_camera_pose = assume_fixed_camera_pose
        self.use_exo_intrinsic_gt = use_exo_intrinsic_gt
        
        # Modify output path based on depth_align_model
        output_path = Path(self.out_cfg.path)
        depth_align_model = self.post_cfg.depth_align_model

        if depth_align_model == "adaptive_unidepth-l_vda":
            output_path = output_path.parent / f"{output_path.name}_static_vda"
        elif depth_align_model == "adaptive_unidepth-l":
            output_path = output_path.parent / f"{output_path.name}_no_vda"
        elif depth_align_model == "adaptive_unidepth-l_metric-vda":
            output_path = output_path.parent / f"{output_path.name}_metric_vda"
        elif depth_align_model == "adaptive_moge_vda":
            output_path = output_path.parent / f"{output_path.name}_moge_static_vda"
        elif depth_align_model == "adaptive_moge":
            output_path = output_path.parent / f"{output_path.name}_moge_no_vda"

        # Add _fixedcam suffix to the output directory name
        if self.assume_fixed_camera_pose:
            output_path = output_path.parent / (output_path.name + "_fixedcam")
        # Add _slammap suffix when save_slam_map is enabled in output config
        if getattr(self.out_cfg, "save_slam_map", False):
            output_path = output_path.parent / (output_path.name + "_slammap")
        # Add _use_gt_intrinsic suffix when GT intrinsics are used
        if self.use_exo_intrinsic_gt is not None:
            output_path = output_path.parent / (output_path.name + "_exo_intr_gt")
        
        self.out_path = output_path
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.camera_type = CameraType(self.init_cfg.camera_type)

    def _add_init_processors(self, video_stream: VideoStream) -> ProcessedVideoStream:
        init_processors: list[StreamProcessor] = []

        # The assertions make sure that the attributes are not estimated previously.
        # Otherwise it will be overwritten by the processors.
        # Skip intrinsics assertion if using GT intrinsics (optimize_intrinsics=False)
        if self.slam_cfg.optimize_intrinsics:
            assert FrameAttribute.INTRINSICS not in video_stream.attributes()
        assert FrameAttribute.CAMERA_TYPE not in video_stream.attributes()
        assert FrameAttribute.METRIC_DEPTH not in video_stream.attributes()
        assert FrameAttribute.INSTANCE not in video_stream.attributes()

        # Use GT intrinsics processor if take_uuid is provided, otherwise use GeoCalib
        if self.use_exo_intrinsic_gt is not None:
            # Parse take_name and start_frame from video path if camera is exo_GT
            take_name = None
            start_frame = None
            
            logger.info(f"Video stream name: {video_stream.name()}")
            logger.info(f"Video stream type: {type(video_stream)}")
            
            if video_stream.name() == "exo_GT":
                # Need to get video path from the underlying stream
                # Unwrap ProcessedVideoStream to get to RawMp4Stream
                current_stream = video_stream
                logger.info(f"Starting unwrap from {type(current_stream)}")
                
                while hasattr(current_stream, 'stream'):
                    current_stream = current_stream.stream
                    logger.info(f"Unwrapped to {type(current_stream)}")
                
                logger.info(f"Final stream type: {type(current_stream)}")
                logger.info(f"Has path attribute: {hasattr(current_stream, 'path')}")
                
                # Now current_stream should be RawMp4Stream which has .path attribute
                if hasattr(current_stream, 'path'):
                    from pathlib import Path
                    video_path = Path(current_stream.path)
                    logger.info(f"Video path: {video_path}")
                    
                    # Get the 4th parent directory name which contains {take_name}_{start_frame}_{end_frame}
                    # Example: uniandes_cooking_008_8_1000_1048
                    take_dir = video_path.parents[3].name
                    logger.info(f"Take dir: {take_dir}")
                    
                    # Parse: split from right by '_', max 2 splits to get [take_name, start_frame, end_frame]
                    parts = take_dir.rsplit('_', 2)
                    logger.info(f"Parts: {parts}")
                    if len(parts) == 3:
                        take_name = parts[0]
                        start_frame = int(parts[1])
                        logger.info(f"Parsed from video path: take_name={take_name}, start_frame={start_frame}")
                    else:
                        logger.warning(f"Failed to parse take_dir '{take_dir}' into 3 parts")
                else:
                    logger.warning(f"Current stream does not have 'path' attribute")
            else:
                logger.info(f"Video stream name is not 'exo_GT', skipping path parsing")
            
            init_processors.append(GTIntrinsicsProcessor(
                video_stream, 
                take_uuid=self.use_exo_intrinsic_gt, 
                camera_type=self.camera_type,
                take_name=take_name,
                start_frame=start_frame
            ))
        else:
            init_processors.append(GeoCalibIntrinsicsProcessor(video_stream, camera_type=self.camera_type))
        if self.init_cfg.instance is not None:
            init_processors.append(
                TrackAnythingProcessor(
                    self.init_cfg.instance.phrases,
                    add_sky=self.init_cfg.instance.add_sky,
                    sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                )
            )
        return ProcessedVideoStream(video_stream, init_processors)

    def _add_post_processors(
        self, view_idx: int, video_stream: VideoStream, slam_output: SLAMOutput
    ) -> ProcessedVideoStream:
        post_processors: list[StreamProcessor] = [
            AssignAttributesProcessor(
                {
                    FrameAttribute.POSE: slam_output.get_view_trajectory(view_idx),  # type: ignore
                    FrameAttribute.INTRINSICS: [slam_output.intrinsics[view_idx]] * len(video_stream),
                }
            )
        ]
        if (depth_align_model := self.post_cfg.depth_align_model) is not None:
            post_processors.append(AdaptiveDepthProcessor(slam_output, view_idx, depth_align_model))
        return ProcessedVideoStream(video_stream, post_processors)

    def run(self, video_data: VideoStream | MultiviewVideoList) -> AnnotationPipelineOutput:
        if isinstance(video_data, MultiviewVideoList):
            video_streams = [video_data[view_idx] for view_idx in range(len(video_data))]
            artifact_paths = [io.ArtifactPath(self.out_path, video_stream.name()) for video_stream in video_streams]
            slam_rig = video_data.rig()

        else:
            assert isinstance(video_data, VideoStream)
            video_streams = [video_data]
            artifact_paths = [io.ArtifactPath(self.out_path, video_data.name())]
            slam_rig = None

        annotate_output = AnnotationPipelineOutput()

        if all([self.should_filter(video_stream.name()) for video_stream in video_streams]):
            logger.info(f"{video_data.name()} has been proccessed already, skip it!!")
            return annotate_output

        slam_streams: list[VideoStream] = [
            # GeoCalibIntrinsicsProcessor로 초기 intrinsics 추정
            self._add_init_processors(video_stream).cache("process", online=True) for video_stream in video_streams
        ]

        slam_pipeline = SLAMSystem(device=torch.device("cuda"), config=self.slam_cfg)
        slam_output = slam_pipeline.run(slam_streams, rig=slam_rig, camera_type=self.camera_type, camera_fix=self.assume_fixed_camera_pose)

        if self.return_payload:
            annotate_output.payload = slam_output
            return annotate_output

        #AssignAttributesProcessor: SLAM 결과를 각 프레임에 할당
        # 카메라 포즈 (6DOF 변환 행렬)
        # 카메라 내재 파라미터
        # AdaptiveDepthProcessor: 적응형 깊이 정렬
        # SVDA (Supervised Video Depth Alignment) 모델 사용
        # 메트릭 스케일 복구
        output_streams = [
            self._add_post_processors(view_idx, slam_stream, slam_output).cache("depth", online=True)
            for view_idx, slam_stream in enumerate(slam_streams)
        ]

        # Dumping artifacts for all views in the streams
        for output_stream, artifact_path in zip(output_streams, artifact_paths):
            artifact_path.meta_info_path.parent.mkdir(exist_ok=True, parents=True)
            if self.out_cfg.save_artifacts:
                logger.info(f"Saving artifacts to {artifact_path}")
                io.save_artifacts(artifact_path, output_stream)
                with artifact_path.meta_info_path.open("wb") as f:
                    pickle.dump({"ba_residual": slam_output.ba_residual}, f)

            if self.out_cfg.save_viz:
                save_projection_video(
                    artifact_path.meta_vis_path,
                    output_stream,
                    slam_output,
                    self.out_cfg.viz_downsample,
                    self.out_cfg.viz_attributes,
                )

            if self.out_cfg.save_slam_map and slam_output.slam_map is not None:
                logger.info(f"Saving SLAM map to {artifact_path.slam_map_path}")
                slam_output.slam_map.save(artifact_path.slam_map_path)

        if self.return_output_streams:
            annotate_output.output_streams = output_streams

        return annotate_output
