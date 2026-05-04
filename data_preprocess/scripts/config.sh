#!/bin/bash
# ============================================================================
# Configuration file for ViPE Inference Pipeline
# ============================================================================
# Modify the values below to customize the pipeline behavior.

# Paths
# REPO_ROOT: Root directory of the repository (auto-detected if not set)
# WORKING_DIR: Output directory (relative to REPO_ROOT or absolute path)
# DATA_DIR: Input data directory (relative to REPO_ROOT or absolute path)
WORKING_DIR="data_preprocess"  # Output directory
DATA_DIR="data_preprocess/example"  # Input data directory (read-only)

# Dataset
EGO4D_DATASET_TYPE="test"  # Dataset type: "test", "train", or "val"

# Frame range
START_FRAME=0
# END_FRAME=$((START_FRAME + 49 - 1))  # Auto-calculated if not set
# Or set directly: END_FRAME=925

# Rendering
POINT_SIZE="5.0"

# Multiprocessing
BATCH_SIZE=3  # Number of parallel processes (recommended: 6-8)

# GT Intrinsic
USE_GT_INTRINSIC=true  # Use ground-truth camera intrinsics from ego_pose annotations

# Advanced settings (usually no need to modify)
UUID_MAPPING_FILE="${WORKING_DIR}/take_name_to_uuid_mapping.json"

# Auto-calculate END_FRAME if not set
if [[ -z "${END_FRAME:-}" ]]; then
    END_FRAME=$((START_FRAME + 49 - 1))
fi

# Auto-detect REPO_ROOT if not set (assumes config.sh is in data_preprocess/scripts/)
if [[ -z "${REPO_ROOT:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

# Convert relative paths to absolute paths
if [[ "$WORKING_DIR" != /* ]]; then
    WORKING_DIR="$(cd "$REPO_ROOT" && cd "$WORKING_DIR" 2>/dev/null && pwd || echo "$REPO_ROOT/$WORKING_DIR")"
fi
if [[ "$DATA_DIR" != /* ]]; then
    DATA_DIR="$(cd "$REPO_ROOT" && cd "$DATA_DIR" 2>/dev/null && pwd || echo "$REPO_ROOT/$DATA_DIR")"
fi

# Validate directories
if [[ ! -d "$WORKING_DIR" ]]; then
    echo "❌ ERROR: WORKING_DIR does not exist: $WORKING_DIR" >&2
    echo "   Please set WORKING_DIR in config.sh to a valid directory" >&2
    exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ ERROR: DATA_DIR does not exist: $DATA_DIR" >&2
    echo "   Please set DATA_DIR in config.sh to a valid directory" >&2
    exit 1
fi
