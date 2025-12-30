# EgoX EgoPrior Rendering from ViPE Results

**This codebase provides tools to generate ego prior videos for `EgoX`. For the EgoX model itself, please refer to the [EgoX](https://github.com/DAVIAN-Robotics/EgoX/) GitHub repository.**

`ViPE` provides point cloud rendering functionality to visualize the 3D reconstruction results. This is particularly useful for analyzing the spatial structure and quality of the estimated depth maps and camera poses.

## 👀 Installation

To ensure the reproducibility, we recommend creating the runtime environment using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

```bash
# Create a new conda environment and install 3rd-party dependencies
conda env create -f envs/base.yml
conda activate egox-egoprior
pip install -r envs/requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" --no-build-isolation
pip install git+https://github.com/microsoft/MoGe.git

# Build the project and install it into the current environment
# Omit the -e flag to install the project as a regular package
pip install --no-build-isolation -e .
```

## 👀 Prerequisites

Before running the rendering commands, ensure you have completed the ViPE inference on your video using the provided script:

```bash
# First, run ViPE inference
./scripts/infer_vipe.sh
```

### ViPE Inference Arguments

The scripts run ViPE inference with various parameters. Below are the key CLI arguments used:

#### Core Arguments

- `--start_frame <int>`: Starting frame number (default: 0)
- `--end_frame <int>`: Ending frame number (inclusive, default: process all frames)
- `--assume_fixed_camera_pose`: Flag to assume camera pose is fixed throughout the video (⚠️ Since `EgoX` is trained on the Ego-Exo4D dataset where exocentric view camera poses are fixed, you must provide exocentric videos with fixed camera poses as input during inference)
- `--pipeline <str>`: Pipeline configuration to use (we used `lyra` for `EgoX`)
  - Available pipelines: `default`, `lyra`, `lyra_no_vda`, `no_vda`, etc.
  - `default`: Uses UniDepthV2 for depth estimation
  - `lyra`: Uses MoGE2 for depth estimation with VDA enabled for better temporal depth consistency
  - `lyra_no_vda` / `no_vda`: Disables Video Depth Anything (VDA) for reduced GPU memory usage

- `--use_exo_intrinsic_gt "<intrinsics_matrix>"`: Use ground truth exocentric camera intrinsics instead of ViPE-estimated intrinsics (e.g., when GT intrinsics are known such as Ego-Exo4D)
  - Takes a 3x3 intrinsics matrix in JSON format: `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
  - Automatically sets `optimize_intrinsics=False` when provided
  - The GT intrinsics are scaled based on current frame resolution (using cy ratio)
  - Example: `--use_exo_intrinsic_gt "[[1000.0,0,960.0],[0,1000.0,540.0],[0,0,1]]"`

### Visualizing ViPE Results

After ViPE inference, you can visualize the results using the built-in visualization tool:

```bash
vipe visualize vipe_results/YOUR_VIPE_RESULT
```

#### Visualization Options

- `--port <int>`: Server port (default: 20540)
- `--use_mean_bg`: Use mean background for visualization (Since EgoX is trained with fixed exocentric camera poses, this option helps visualize cleaner point clouds for static objects)
- `--ego_manual`: **Enable manual ego trajectory annotation mode**. Use this option when you want to obtain ego trajectory directly from in-the-wild videos. 
  
  **Manual annotation workflow:**
  1. For each frame, position the ego camera frustum to align with the appropriate head pose in the 3D view
  2. Fill in the `ego_extrinsics` field in `meta.json` using the ego camera extrinsics displayed in the top-right UI panel."
  3. Repeat for all frames to build the complete ego trajectory
  4. See **Appendix Fig. 8** in the paper for examples of frustum positioning aligned with head poses

  **Important Note for In-the-Wild Videos:**
  
  Since ego trajectories are manually annotated for in-the-wild videos, the final rendering results can vary significantly depending on how you position the ego camera frustums. Different annotation strategies may lead to different visual perspectives in the rendered ego-view videos.

  Below is a comparison showing **EgoX generation results from two different ego trajectory annotations** for the same exocentric input video (Ironman scene).

  | Version 1 | Version 2 |
  |-----------|-----------|
  | <img src="assets/ironman_ver1.png" width="400"> | <img src="assets/ironman_ver2.png" width="400"> |

The visualization tool provides an interactive 3D viewer where you can:
- Inspect point clouds and camera poses
- Validate depth map quality
- Manually annotate ego trajectories for in-the-wild videos (with `--ego_manual` flag)

## 👀 Ego Prior Rendering

For convenient batch processing, use the provided rendering script:

```bash
./scripts/render_vipe.sh
```

This script executes the point cloud rendering with the following configuration:

- `--input_dir`: ViPE inference results directory
- `--out_dir`: Output directory for rendered results
- `--meta_json_path`: JSON file which include camera parameters
- `--point_size`: Point cloud visualization size
- `--start_frame`/`--end_frame`: Frame range (both inclusive)
- `--fish_eye_rendering`: Enables fish-eye distortion rendering
- `--use_mean_bg`: Uses mean background for rendering
- `--only_bg`: Renders only the background point clouds (exclude dynamic instance's point clouds)

### Camera Parameters Format

The `meta.json` file should contain camera intrinsics and extrinsics in the following format:

```json
{
  "test_datasets": [
    {
      "exo_path": "./example/in_the_wild/videos/joker/exo.mp4",
      "ego_prior_path": "./example/in_the_wild/videos/joker/ego_Prior.mp4",
      "camera_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "camera_extrinsics": [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]],
      "ego_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
      "ego_extrinsics": [
        [[r11, r12, r13, tx], [r21, r22, r23, ty], [r31, r32, r33, tz]],
        ...
      ]
    }
  ]
}
```

All extrinsics matrices are in world-to-camera format (3x4). The script will automatically convert them to 4x4 format by adding `[0, 0, 0, 1]` as the last row.

### Manual Rendering Command

For manual execution or custom configurations, you can also run the rendering script directly:

```bash
python scripts/render_vipe_pointcloud.py \
  --input_dir vipe_results/YOUR_VIDEO_NAME \
  --meta_json_path /path/to/meta.json \
  --out_dir /path/to/output \
  --start_frame 0 \
  --end_frame 100 \
  --point_size 5.0 \
  --fish_eye_rendering \
  --use_mean_bg
```

### Output Structure

The rendered results will be saved as MP4 videos (30 FPS) in the following structure:
```
example/egoexo4D/videos/
├── cmu_soccer_06_6_877_925/
│   ├── ego_Prior.mp4
│   └── exo.mp4
├── iiith_cooking_57_2_2451_2499/
│   ├── ego_Prior.mp4
│   └── exo.mp4
├── sfu_basketball014_4_1000_1048/
│   ├── ego_Prior.mp4
│   └── exo.mp4
└── ...
```

Each result is saved in a directory named after the input ViPE result (e.g., `vipe_results/joker` → `joker/ego_prior.mp4`).

### Example of Ego Prior Rendering

  <img src="assets/ironman_egoprior.png" width="400">

## 👀 Converting Depth Maps for EgoX Model

After ViPE inference, you need to convert the depth maps from `.zip` archives (containing `.exr` files) to `.npy` format that the EgoX model can process:

```bash
python scripts/convert_depth_zip_to_npy.py \
  --depth_path {EgoX_path}/vipe_results/YOUR_VIDEO/depth \
  --egox_depthmaps_path {EgoX_path}/example/egoexo4D/depth_maps
```

This script will:
- Extract all `.exr` depth maps from the zip archive(s) in the specified directory
- Convert them to `.npy` format
- Save them to `{egox_depthmaps_path}/{zip_filename}/` directory structure

**Note**: This conversion step is independent of EgoPrior rendering and is specifically required as a preprocessing step before feeding data into the EgoX model.

## Performance Tips
- **Tuning ViPE inference**: You can adjust temporal and spatial consistency in ViPE inference results by:
  - Changing the underlying models used internally by ViPE (e.g., switching depth estimation models)
  - Adjusting model sizes (e.g., using larger models for better quality or smaller models for faster processing)
  - Modifying pipeline configurations to balance between temporal consistency and 3D spatial consistency
- Use the visualization tools (`vipe visualize`) to preview results before running extensive rendering jobs
- The rendering quality depends on the depth estimation quality from the original ViPE inference

## 👀 EgoExo4D Training Data Preprocessing

For pre-processing of EgoExo4D data for training EgoX, we provide a comprehensive preprocessing pipeline that automates ViPE inference and ego prior rendering for multiple takes.

### Data Structure

The preprocessing pipeline expects the following directory structure:

```
your_data_directory/
├── takes/
│   ├── take_name_1/
│   │   └── frame_aligned_videos/
│   │       └── downscaled/
│   │           └── 448/
│   │               ├── cam01.mp4
│   │               ├── cam02.mp4
│   │               └── ...
│   └── take_name_2/
│       └── ...
├── annotations/
│   └── ego_pose/
│       └── test/
│           └── camera_pose/
│               ├── uuid_1.json
│               ├── uuid_2.json
│               └── ...
└── captures.json
```

**Example Data**: See `data_preprocess/example/` for a minimal example of the required data structure with 3 sample takes.

### Configuration

1. **Edit the configuration file** (`data_preprocess/scripts/config.sh`):

```bash
# Paths
WORKING_DIR="/path/to/your/output/directory"  # Output directory
DATA_DIR="/path/to/your/egoexo4d/data"        # Input data directory (read-only)

# Frame range
START_FRAME=0
END_FRAME=49  # Or auto-calculated: END_FRAME=$((START_FRAME + 49 - 1))

# Rendering
POINT_SIZE="5.0"

# Multiprocessing
BATCH_SIZE=6  # Number of parallel processes (recommended: 6-8)
```

2. **Key Configuration Parameters**:
   - `WORKING_DIR`: Directory where all output files (ViPE results, rendered videos, metadata) will be saved
   - `DATA_DIR`: Path to your EgoExo4D dataset directory containing `takes/`, `annotations/`, and `captures.json`
   - `START_FRAME` / `END_FRAME`: Frame range to process (default: 0-48 for 49 frames)
   - `BATCH_SIZE`: Number of takes to process in parallel

### Running the Preprocessing Pipeline

After configuring `config.sh`, run the batch processing script:

```bash
cd /path/to/EgoX-EgoPriorRenderer
bash data_preprocess/scripts/infer_vipe_all_takes.sh
```

The script will:
1. **Load all takes** from `DATA_DIR/takes/`
2. **Run ViPE inference** for each camera in each take (using the `lyra` pipeline)
3. **Generate `meta.json` files** automatically from `ego_pose` annotations
4. **Render ego prior videos** for each camera
5. **Select the best camera** based on rendering quality metrics
6. **Save final results** to `WORKING_DIR/data/{START_FRAME}_{END_FRAME}/best_ego_view_rendering/`

### Output Structure

The preprocessing pipeline generates the following output structure:

```
WORKING_DIR/
├── data/
│   └── {START_FRAME}_{END_FRAME}/
│       ├── best_ego_view_rendering/
│       │   ├── take_name_1/
│       │   │   ├── ego_Prior/
│       │   │   │   └── ego_Prior.mp4
│       │   │   ├── exo_GT/
│       │   │   │   └── frame_*.png
│       │   │   ├── ego_GT/
│       │   │   │   └── frame_*.png
│       │   │   └── metadata.json
│       │   └── take_name_2/
│       │       └── ...
│       ├── vipe_results/
│       │   └── take_name_1/
│       │       └── camera_result_subdir/
│       │           ├── pose/
│       │           ├── rgb/
│       │           ├── depth/
│       │           └── ...
│       └── meta_files/
│           └── meta_take_name_result_subdir.json
└── take_name_to_uuid_mapping.json
```

### Advanced Options

You can also specify batch size via command-line arguments:

```bash
bash data_preprocess/scripts/infer_vipe_all_takes.sh --batch-size 8
```

### Notes

- The script automatically creates a UUID mapping file (`take_name_to_uuid_mapping.json`) from `ego_pose` annotations if it doesn't exist
- Processing can be resumed: the script skips takes that already have completed results in `best_ego_view_rendering/`
- Error logs are saved to `WORKING_DIR/data/{START_FRAME}_{END_FRAME}/.error/` for debugging
- The best camera selection is based on rendering quality metrics (frames with white pixels, total white pixels)

## 🙏 Acknowledgements

This `EgoX`'s ego prior rendering codebase is built upon the `ViPE`(Video Pose Engine) project. We gratefully acknowledge their excellent work in video pose estimation and depth map generation. For more details, please visit the [ViPE](https://github.com/nv-tlabs/vipe) GitHub repository.

