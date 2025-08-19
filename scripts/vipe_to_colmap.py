#!/usr/bin/env python3

import argparse
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import Imath
import numpy as np
import OpenEXR
from scipy.spatial.transform import Rotation

from vipe.utils.io import ArtifactPath

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    rotation = Rotation.from_matrix(matrix[:3, :3])
    quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # Convert to [w, x, y, z]


def matrix_to_colmap_pose(c2w_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert camera-to-world matrix to COLMAP format.
    COLMAP uses world-to-camera transformation.
    """
    # Convert camera-to-world to world-to-camera
    w2c = np.linalg.inv(c2w_matrix)

    # Extract rotation and translation
    rotation_matrix = w2c[:3, :3]
    translation = w2c[:3, 3]

    # Convert rotation matrix to quaternion
    quaternion = quaternion_from_matrix(w2c)

    return quaternion, translation

def exr_to_depth(exr_file) -> np.ndarray:
    """Extract depth information from an EXR file."""
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Try reading "Z" channel
    depth_str = exr_file.channel("Z", pt)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)

    return depth

def write_cameras_txt(output_dir: Path, artifact: ArtifactPath, frame_width: int, frame_height: int):
    """Write COLMAP cameras.txt file."""
    cameras_file = output_dir / "cameras.txt"

    # Load intrinsics data
    intrinsics_data = np.load(artifact.intrinsics_path)
    intrinsics = intrinsics_data['data']  # Shape: (N, 4) -> [fx, fy, cx, cy]

    with open(cameras_file, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")

        # Use first frame's intrinsics (assuming constant intrinsics)
        fx, fy, cx, cy = intrinsics[0]

        # COLMAP camera format: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
        f.write(f"1 PINHOLE {frame_width} {frame_height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    logger.info(f"Written cameras.txt with intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


def write_images_txt(output_dir: Path, artifact: ArtifactPath):
    """Write COLMAP images.txt file."""
    images_file = output_dir / "images.txt"

    # Load pose data
    pose_data = np.load(artifact.pose_path)
    poses = pose_data['data']  # Shape: (N, 4, 4)
    indices = pose_data['inds']  # Frame indices

    with open(images_file, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(poses)}\n")

        for i, (pose_matrix, frame_idx) in enumerate(zip(poses, indices)):
            # Convert pose to COLMAP format
            quaternion, translation = matrix_to_colmap_pose(pose_matrix)
            qw, qx, qy, qz = quaternion
            tx, ty, tz = translation

            # Image filename
            image_name = f"images/frame_{frame_idx:06d}.jpg"

            # Write image line
            f.write(f"{i+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {tx:.9f} {ty:.9f} {tz:.9f} 1 {image_name}\n")
            # Empty points2D line (no 2D-3D correspondences)
            f.write("\n")

    logger.info(f"Written images.txt with {len(poses)} images")


def depth_to_point_cloud(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                        c2w: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Convert depth map to 3D point cloud in world coordinates."""
    height, width = depth.shape

    # Create pixel coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to homogeneous coordinates and filter valid depth
    valid_mask = (depth > 0) & (depth < np.inf) & ~np.isnan(depth)

    # Get valid pixels
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    depth_valid = depth[valid_mask]

    # Convert pixel coordinates to camera coordinates
    x_cam = (u_valid - cx) * depth_valid / fx
    y_cam = (v_valid - cy) * depth_valid / fy
    z_cam = depth_valid

    # Stack to homogeneous coordinates [x, y, z, 1]
    points_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(x_cam)], axis=0)

    # Transform to world coordinates using c2w matrix
    points_world = c2w @ points_cam
    points_world = points_world[:3]  # Remove homogeneous coordinate

    # Extract RGB colors for valid points
    rgb_valid = rgb[valid_mask]

    # Combine 3D points with RGB colors
    points_with_color = np.column_stack([points_world.T, rgb_valid])

    return points_with_color


def write_points3d_txt(output_dir: Path, artifact: ArtifactPath, depth_step=1):
    """Write empty COLMAP points3D.txt file."""
    depth_data = []
    with zipfile.ZipFile(artifact.depth_path, 'r') as zf:
        depth_files = [name for name in sorted(zf.namelist()) if name.endswith('.exr')]
        for depth_file in depth_files:
            with zf.open(depth_file) as df:
                depth = exr_to_depth(OpenEXR.InputFile(df))
                depth_data.append(depth)
    pose_data = np.load(artifact.pose_path)
    intrinsics_data = np.load(artifact.intrinsics_path)
    points3d_file = output_dir / "points3D.txt"

    image_dir = output_dir / "images"; images = sorted(list(image_dir.glob("*.jpg")))
    # Collect all 3D points first
    all_points = []
    point_id = 1

    for i, depth in enumerate(depth_data):
        if i % depth_step != 0:
            continue
        fx, fy, cx, cy = intrinsics_data['data'][i]
        c2w = pose_data['data'][i]
        rgb = cv2.cvtColor(cv2.imread(str(images[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        points_3d = depth_to_point_cloud(depth, fx, fy, cx, cy, c2w, rgb)

        # Add image index and point ID to each point
        # skip every N points
        for point in points_3d[::16]:
            x, y, z, r, g, b = point
            all_points.append((point_id, x, y, z, int(r), int(g), int(b), 0.0, i + 1))
            point_id += 1

        if i % 30 == 0:
            logger.info(f"Processed {i}/{len(depth_data)} depth maps")

    # Write points to file
    with open(points3d_file, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(all_points)}\n")

        for point_data in all_points:
            point_id, x, y, z, r, g, b, error, image_id = point_data
            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f} 0 {image_id} 0 0 0 0\n")

    logger.info(f"Written points3D.txt with {len(all_points)} points")


def extract_frames(artifact: ArtifactPath, output_dir: Path) -> Tuple[int, int]:
    """Extract frames from video to individual image files."""
    video_path = artifact.rgb_path
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    logger.info(f"Extracting frames from {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save frame as JPEG
        frame_filename = f"frame_{frame_idx:06d}.jpg"
        frame_path = images_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

        frame_idx += 1

        if frame_idx % 30 == 0:
            logger.info(f"Extracted {frame_idx}/{frame_count} frames")

    cap.release()
    logger.info(f"Extracted {frame_idx} frames to {images_dir}")

    return frame_width, frame_height


def convert_vipe_to_colmap(vipe_path: Path, output_path: Path, depth_step: int):
    """Convert ViPE reconstruction results to COLMAP format."""

    logger.info(f"Converting ViPE results from {vipe_path} to COLMAP format at {output_path}")

    # Find artifacts
    artifacts = list(ArtifactPath.glob_artifacts(vipe_path, use_video=True))

    if len(artifacts) == 0:
        raise ValueError(f"No artifacts found in {vipe_path}")

    # Select first available artifact (by default)
    artifact = artifacts[0]
    logger.info(f"Using artifact: {artifact.artifact_name}")

    # Verify required files exist
    required_files = [artifact.rgb_path, artifact.pose_path, artifact.intrinsics_path, artifact.depth_path]
    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract frames and get video dimensions
    frame_width, frame_height = extract_frames(artifact, output_path)

    # Write COLMAP files
    write_cameras_txt(output_path, artifact, frame_width, frame_height)
    write_images_txt(output_path, artifact)
    write_points3d_txt(output_path, artifact, depth_step)

    # Copy or link original video for reference
    reference_video = output_path / f"original_{artifact.artifact_name}.mp4"
    shutil.copy2(artifact.rgb_path, reference_video)

    logger.info(f"COLMAP conversion completed successfully!")
    logger.info(f"Output directory: {output_path}")
    logger.info(f"Files created:")
    logger.info(f"  - cameras.txt: Camera intrinsics")
    logger.info(f"  - images.txt: Camera poses ({len(artifacts)} images)")
    logger.info(f"  - points3D.txt: 3D points")
    logger.info(f"  - images/: Individual frame images")


def main():
    """Main function for ViPE to COLMAP conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert ViPE reconstruction results to COLMAP format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "vipe_path",
        type=Path,
        help="Path to ViPE results directory"
    )
    parser.add_argument(
        "--depth_step",
        type=int,
        default=4,
        help="Step size for depth extraction (default: 4)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output directory for COLMAP format (default: <vipe_path>_colmap)"
    )

    args = parser.parse_args()

    if not args.vipe_path.exists():
        print(f"Error: ViPE path '{args.vipe_path}' does not exist.")
        return 1

    # Set default output path
    if args.output is None:
        args.output = args.vipe_path.parent / f"{args.vipe_path.name}_colmap"

    try:
        convert_vipe_to_colmap(args.vipe_path, args.output, args.depth_step)
        return 0
    except Exception as e:
        logger.exception("Conversion failed:")
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())