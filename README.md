# Ego View Rendering from ViPE Results

ViPE provides point cloud rendering functionality to visualize the 3D reconstruction results. This is particularly useful for analyzing the spatial structure and quality of the estimated depth maps and camera poses.

### Prerequisites

Before running the rendering commands, ensure you have completed the ViPE inference on your video using the provided script:

```bash
# First, run ViPE inference using the batch script
./scripts/infer_vipe.sh
```

This script will run ViPE inference with the following parameters: `--start_frame`, `--end_frame`, `--assume_fixed_camera_pose`, and `--pipeline` settings. The script generates the necessary pose and depth information in the output directory.

### Rendering with Scripts

For convenient batch processing, use the provided rendering script:

```bash
./scripts/render_vipe.sh
```

This script executes the point cloud rendering with multiple parameters configured for Ego4D dataset processing:

- `--input_dir`: ViPE inference results directory
- `--out_dir`: Output directory for rendered results
- `--ego_camera_pose_path`: Ego camera pose JSON file
- `--exo_camera_pose_path`: Exocentric camera calibration CSV
- `--online_calibration_path`: Online calibration data
- `--point_size`: Point cloud visualization size
- `--start_frame`/`--end_frame`: Frame range (both inclusive)
- `--fish_eye_rendering`: Enables fish-eye distortion rendering
- `--use_mean_bg`: Uses mean background for rendering
- `--only_bg`: Renders only the background point clouds (exclude dynamic instance's point clouds)

The script also supports multi-GPU parallel processing and can be configured by modifying the experiment settings within the script.

### Manual Rendering Command

For manual execution or custom configurations, you can also run the rendering script directly:

```bash
python ego_view_rendering/render_vipe_pointcloud.py \
  --input_dir vipe_results/YOUR_VIDEO_NAME \
  --start_frame 0 \
  --end_frame 100 \
  --out_dir ego_view_rendering \
  --point_size 1.5 \
  --fish_eye_rendering \
  --use_mean_bg
```

### Output Structure

The rendered results will be saved in the following structure:
```
ego_view_rendering/
├── cmu_bike01_2/
│   └── cam02/
│       ├── test_output_pts1.0/
│       ├── test_output_static_vda_fixedcam_fisheye_pts1.0/
│       ├── test_output_static_vda_fixedcam_mean_bg_fisheye_pts1.5/
│       └── test_output_static_vda_fixedcam_slammap_fisheye_pts0.5/
│       ```
├── fair_cooking_05_2/
├── georgiatech_cooking_01_01_2/
├── iiith_cooking_01_1/
├── indiana_cooking_01_2/
├── minnesota_cooking_010_2/
├── nus_cooking_06_2/
├── sfu_cooking015_2/
└── uniandes_cooking_001_10/
```

Each experiment directory contains multiple output subdirectories with different rendering configurations (pipeline type, point size, background settings, etc.).

### Common Issues and Solutions

**Issue**: `ValueError: No valid camera ID found in directory`
- **Solution**: Ensure your inference results contain properly formatted camera data

**Issue**: `Frame range [X, Y] exceeds available inference results [START_FRAME, END_FRAME]`
- **Solution**: Check the available frame range in your inference results (which corresponds to the start_frame and end_frame used during inference) and adjust `--start_frame` and `--end_frame` accordingly

**Issue**: Missing pose or depth data
- **Solution**: Verify that ViPE inference completed successfully and generated all necessary output files (`pose/*.npz`, `depth/*.npy`)

### Performance Tips

- For large videos, consider processing smaller frame ranges to reduce memory usage and processing time
- The rendering quality depends on the depth estimation quality from the original ViPE inference
- Use the visualization tools (`vipe visualize`) to preview results before running extensive rendering jobs

<br/><br/><br/>


<br/>

# ViPE: Video Pose Engine for Geometric 3D Perception

<p align="center">
  <img src="assets/teaser.gif" alt="teaser"/>
</p>

**TL;DR: ViPE is a useful open-source spatial AI tool for annotating camera poses and dense depth maps from raw videos!**

**Contributors**: NVIDIA (Spatial Intelligence Lab, Dynamic Vision Lab, NVIDIA Issac, NVIDIA Research).

**Full Abstract**: Accurate 3D geometric perception is an important prerequisite for a wide range of spatial AI systems. While state-of-the-art methods depend on large-scale training data, acquiring consistent and precise 3D annotations from in-the-wild videos remains a key challenge. In this work, we introduce ViPE, a handy and versatile video processing engine designed to bridge this gap. ViPE efficiently estimates camera intrinsics, camera motion, and dense, near-metric depth maps from unconstrained raw videos. It is robust to diverse scenarios, including dynamic selfie videos, cinematic shots, or dashcams, and supports various camera models such as pinhole, wide-angle, and 360° panoramas. 
We use ViPE to annotate a large-scale collection of videos. This collection includes around 100K real-world internet videos, 1M high-quality AI-generated videos, and 2K panoramic videos, totaling approximately 96M frames -- all annotated with accurate camera poses and dense depth maps. We open source ViPE and the annotated dataset with the hope to accelerate the development of spatial AI systems.

**[Technical Whitepaper](https://research.nvidia.com/labs/toronto-ai/vipe/assets/paper.pdf), [Project Page](https://research.nvidia.com/labs/toronto-ai/vipe), [Dataset](#downloading-the-dataset)**

## Installation

To ensure the reproducibility, we recommend creating the runtime environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

```bash
# Create a new conda environment and install 3rd-party dependencies
conda env create -f envs/base.yml
conda activate vipe
pip install -r envs/requirements.txt

# Build the project and install it into the current environment
# Omit the -e flag to install the project as a regular package
pip install --no-build-isolation -e .
```

## Usage

### Using the ViPE CLI

Once the python package is installed, you can use the `vipe` CLI to process raw videos in mp4 format.

```bash
# Replace YOUR_VIDEO.mp4 with the path to your video. We provide sample videos in assets/examples.
vipe infer YOUR_VIDEO.mp4
# Additional options:
#   --output: Output directory (default: vipe_results)
#   --visualize: Enable visualization of intermediate and final results (default: false)
#   --pipeline: Pipeline configuration to use (default: default)
```

![vipe-vis](assets/vipe-vis.gif)

One can visualize the results that ViPE produces by running (supported by `viser`):
```bash
vipe visualize vipe_results/
# Please modify the above vipe_results/ path to the output directory of your choice.
```

![vipe-viser](assets/vipe-viser.gif)

> We found that running [video-depth-anything](https://github.com/DepthAnything/Video-Depth-Anything) might eat up too much of GPU memory. To that end we provide a `no_vda` config that produces less temporally-stable depth (but empirically more 3D consistent) maps. This can be triggered by adding `--pipeline no_vda` to the `vipe infer` command.

### Using the `run.py` script

The `run.py` script is a more flexible way to run ViPE. Compared to the CLI, the script supports running on multiple videos at once and allows more fine-grained control over the pipeline with `hydra` configs. It also provides an example of using `vipe` as a library in your own project.

Example usages:

```bash
# Running the full pipeline.
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH

# Running the pose-only pipeline without depth estimation.
python run.py pipeline=default streams=raw_mp4_stream streams.base_path=YOUR_VIDEO_OR_DIR_PATH pipeline.post.depth_align_model=null
```

### Converting to COLMAP format

You can use the following script to convert the ViPE results to COLMAP format. For example:
```bash
python scripts/vipe_to_colmap.py vipe_results/ --sequence dog_example
```
This will unproject the dense depth maps to create the 3D point cloud. 
Alternatively for a more lightweight and 3D consistent point cloud, you can add the `--use_slam_map` flag to the above command. This requires you to run the full pipeline with `pipeline.output.save_slam_map=true` to save the additional information.

## Downloading the Dataset

![dataset](assets/dataset.gif)

Together with ViPE we release a large-scale dataset containing ~1M high-quality videos with accurate camera poses and dense depth maps. Specifications of the datasets are listed below:

| Dataset Name   | # Videos | # Frames | Hugging Face Link                                            | License      | Prefix |
| -------------- | -------- | -------- | ------------------------------------------------------------ | ------------ | ------ |
| Dynpose-100K++ | 99,501   | 15.8M    | [Link](https://huggingface.co/datasets/nvidia/vipe-dynpose-100kpp) | CC-BY-NC 4.0 | `dpsp` |
| Wild-SDG-1M    | 966,448  | 78.2M    | [Link](https://huggingface.co/datasets/nvidia/vipe-wild-sdg-1m) | CC-BY-NC 4.0 | `wsdg` |
| Web360         | 2,114    | 212K     | [Link](https://huggingface.co/datasets/nvidia/vipe-web360)   | CC-BY 4.0    | `w360` |

You can download the datasets using the following utility script:

```bash
# Replace YOUR_PREFIX with the prefix of the dataset to be downloaded (see prefix column in the table above)
# You can also use more specific prefixes, e.g. wsdg-003e2c86 to download a specific shard of the dataset.
python scripts/download_dataset.py --prefix YOUR_PREFIX --output_base YOUR_OUTPUT_DIR --rgb --depth
```

> Note that the depth component is very large and you might expect a long downloading time. For `rgb` component of the Dynpose-100K++ dataset, we directly retrieve the RGB frames from YouTube. You have to `pip install yt_dlp ffmpeg-python` to use this feature. Please refer to the original [Dynpose-100K dataset](https://huggingface.co/datasets/nvidia/dynpose-100k) for alternative approaches to retrieve the videos.

The dataset itself can be visualized using the same visualization script:
```bash
vipe visualize YOUR_OUTPUT_DIR
```

## Acknowledgments

ViPE is built on top of many great open-source research projects and codebases. Some of these include (not exhaustive):
- [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Metric3Dv2](https://github.com/YvanYin/Metric3D)
- [PriorDA](https://github.com/SpatialVision/Prior-Depth-Anything)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
- [VideoDepthAnything](https://github.com/DepthAnything/Video-Depth-Anything)
- [GeoCalib](https://github.com/cvg/GeoCalib)
- [Segment and Track Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)

Please refer to the [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for a full list of projects and their licenses.

We thank useful discussions from Aigul Dzhumamuratova, Viktor Kuznetsov, Soha Pouya, and Ming-Yu Liu, as well as release support from Vishal Kulkarni.

## TODO

- [x] Initial code released under Apache 2.0 license.
- [x] Full dataset uploaded to Hugging Face for download.
- [ ] Add instructions to run inference on wide-angle and 360° videos.
- [ ] Add instructions for benchmarking.

## Citation

If you find ViPE useful in your research or application, please consider citing the following whitepaper:

```
@inproceedings{huang2025vipe,
    title={ViPE: Video Pose Engine for 3D Geometric Perception},
    author={Huang, Jiahui and Zhou, Qunjie and Rabeti, Hesam and Korovko, Aleksandr and Ling, Huan and Ren, Xuanchi and Shen, Tianchang and Gao, Jun and Slepichev, Dmitry and Lin, Chen-Hsuan and Ren, Jiawei and Xie, Kevin and Biswas, Joydeep and Leal-Taixe, Laura and Fidler, Sanja},
    booktitle={NVIDIA Research Whitepapers arXiv:2508.10934},
    year={2025}
}
```

## License

This project will download and install additional third-party **models and softwares**. Note that these models or softwares are not distributed by NVIDIA. Review the license terms of these models and projects before use. This source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).
