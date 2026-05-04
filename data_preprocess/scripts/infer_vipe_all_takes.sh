#!/bin/bash

# Multiprocessing setup
set -uo pipefail
# set -e removed (don't exit immediately on error)

# ============================================================================
# Load configuration file
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.sh"

if [[ -f "$CONFIG_FILE" ]]; then
    echo "📖 Loading configuration from: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "❌ ERROR: Config file not found: $CONFIG_FILE" >&2
    echo "   Please create $CONFIG_FILE with required settings" >&2
    echo "   See config.sh.example for a template" >&2
    exit 1
fi

# Validate REPO_ROOT
if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "❌ ERROR: REPO_ROOT is not set" >&2
    echo "   Please set REPO_ROOT in config.sh or ensure it can be auto-detected" >&2
    exit 1
fi

# Validate EGO4D_DATASET_TYPE
if [[ -z "${EGO4D_DATASET_TYPE:-}" ]]; then
    echo "⚠️  WARNING: EGO4D_DATASET_TYPE is not set, defaulting to 'test'" >&2
    EGO4D_DATASET_TYPE="test"
fi

# ============================================================================
# Derived paths (based on loaded config)
# ============================================================================

TAKES_DIR="${DATA_DIR}/takes"
CAPTURES_FILE="${DATA_DIR}/captures.json"
EGO_POSE_DIR="${DATA_DIR}/annotations/ego_pose/test/camera_pose"

BASE_PATH="${TAKES_DIR}"
DATA_ROOT="${WORKING_DIR}/data/${START_FRAME}_${END_FRAME}"
OUTPUT_DIR="${DATA_ROOT}/ego_view_rendering"
BEST_OUT_DIR="${DATA_ROOT}/best_ego_view_rendering"
VIPE_RESULTS_ROOT="${DATA_ROOT}/vipe_results"
DEFAULT_VIPE_ROOT="${WORKING_DIR}/vipe_results"
PROGRESS_ROOT="${DATA_ROOT}/.progress"
ERROR_ROOT="${DATA_ROOT}/.error"
META_FILES_DIR="${DATA_ROOT}/meta_files"
TOTAL_TAKES=0
PIDS=()  # Running process IDs
TEMP_DIR="/tmp/vipe_multiprocessing_$$"

mkdir -p "$DATA_ROOT" "$OUTPUT_DIR" "$BEST_OUT_DIR" "$VIPE_RESULTS_ROOT" "$PROGRESS_ROOT" "$ERROR_ROOT" "$META_FILES_DIR" "$TEMP_DIR"

# Signal handling
cleanup_and_exit() {
    echo ""
    echo "🛑 Interrupt signal received. Cleaning up all processes..."
    
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Terminating process $pid..."
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Force killing process $pid..."
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    wait 2>/dev/null || true
    rm -rf "$TEMP_DIR" 2>/dev/null || true
    
    echo "✅ Cleanup complete. Exiting."
    exit 130
}

trap cleanup_and_exit SIGINT SIGTERM

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -b, --batch-size SIZE    Number of parallel processes (default: 1)"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --batch-size 4        # Run with 4 parallel processes"
    echo "  $0 -b 2                  # Run with 2 parallel processes"
    exit 1
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--batch-size)
                BATCH_SIZE="$2"
                if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [[ "$BATCH_SIZE" -lt 1 ]]; then
                    echo "❌ Error: Batch size must be an integer >= 1"
                    exit 1
                fi
                shift 2
                ;;
            -h|--help)
                usage
                ;;
            *)
                echo "❌ Unknown option: $1"
                usage
                ;;
        esac
    done
}

cleanup_zombies() {
    while true; do
        wait -n 2>/dev/null || true
        sleep 1
    done
}

load_all_takes() {
    local takes_dir="$1"
    local all_takes=()
    
    if [[ ! -d "$takes_dir" ]]; then
        echo "❌ ERROR: Takes directory not found: $takes_dir" >&2
        return 1
    fi
    
    for take_dir in "$takes_dir"/*; do
        if [[ -d "$take_dir" ]]; then
            take_name=$(basename "$take_dir")
            all_takes+=("$take_name")
        fi
    done
    
    IFS=$'\n' sorted_takes=($(sort <<<"${all_takes[*]}"))
    unset IFS
    
    echo "📖 Loading all takes from: $takes_dir" >&2
    echo "   Loaded ${#sorted_takes[@]} takes from directory" >&2
    
    printf '%s\n' "${sorted_takes[@]}"
}

collect_best_processed_takes() {
    shopt -s nullglob
    for d in "$BEST_OUT_DIR"/*; do
        if [[ -d "$d" ]]; then
            basename "$d"
        fi
    done
    shopt -u nullglob
}

find_last_completed_take() {
    local takes_list=("$@")
    local best_takes=()
    
    while IFS= read -r tname; do
        best_takes+=("$tname")
    done < <(collect_best_processed_takes)
    
    local last_index=-1
    for i in "${!takes_list[@]}"; do
        local take_name="${takes_list[$i]}"
        for best_take in "${best_takes[@]}"; do
            if [[ "$take_name" == "$best_take" ]]; then
                if [[ $i -gt $last_index ]]; then
                    last_index=$i
                fi
                break
            fi
        done
    done
    
    if [[ $last_index -ge 0 ]]; then
        echo "${takes_list[$last_index]}"
    else
        echo ""
    fi
}

collect_output_processed_takes() {
    shopt -s nullglob
    for d in "$OUTPUT_DIR"/*; do
        if [[ -d "$d" ]]; then
            basename "$d"
        fi
    done
    shopt -u nullglob
}

count_total_takes() {
    local count=0
    for take_dir in "${TAKES_DIR}"/*; do
        if [[ -d "$take_dir" ]]; then
            ((count++))
        fi
    done
    echo "$count"
}

process_single_take() {
    local take_dir="$1"
    local take_name="$2"
    local uuid="$3"
    local batch_id="$4"
    local log_file="$TEMP_DIR/take_${take_name}_batch_${batch_id}.log"
    
    {
        echo "[BATCH-$batch_id] Processing take: $take_name (UUID: $uuid)"
        
        camera_files=($(get_camera_files "$take_dir"))
        
        if [[ ${#camera_files[@]} -eq 0 ]]; then
            echo "[BATCH-$batch_id] ERROR: No camera files found for take: $take_name"
            echo "[BATCH-$batch_id] Take directory: $take_dir"
            echo "[BATCH-$batch_id] Expected video directory: ${take_dir}/frame_aligned_videos/downscaled/448"
            echo "[BATCH-$batch_id] SKIPPING take: $take_name"
            
            local error_file="$ERROR_ROOT/${take_name}_batch_${batch_id}.error"
            {
                echo "ERROR: No camera files found for take: $take_name"
                echo "Timestamp: $(date)"
                echo "Batch ID: $batch_id"
                echo "Process ID: $$"
                echo "Take directory: $take_dir"
                echo "Expected video directory: ${take_dir}/frame_aligned_videos/downscaled/448"
                echo "UUID: $uuid"
                echo ""
                echo "Directory structure check:"
                if [[ -d "$take_dir" ]]; then
                    echo "  ✓ Take directory exists: $take_dir"
                    ls -la "$take_dir" 2>/dev/null || echo "  ✗ Cannot list take directory contents"
                else
                    echo "  ✗ Take directory does not exist: $take_dir"
                fi
                echo ""
                if [[ -d "${take_dir}/frame_aligned_videos" ]]; then
                    echo "  ✓ frame_aligned_videos directory exists"
                    ls -la "${take_dir}/frame_aligned_videos" 2>/dev/null || echo "  ✗ Cannot list frame_aligned_videos contents"
                else
                    echo "  ✗ frame_aligned_videos directory does not exist"
                fi
                echo ""
                if [[ -d "${take_dir}/frame_aligned_videos/downscaled" ]]; then
                    echo "  ✓ downscaled directory exists"
                    ls -la "${take_dir}/frame_aligned_videos/downscaled" 2>/dev/null || echo "  ✗ Cannot list downscaled contents"
                else
                    echo "  ✗ downscaled directory does not exist"
                fi
                echo ""
                if [[ -d "${take_dir}/frame_aligned_videos/downscaled/448" ]]; then
                    echo "  ✓ 448 directory exists"
                    echo "  Files in 448 directory:"
                    ls -la "${take_dir}/frame_aligned_videos/downscaled/448" 2>/dev/null || echo "  ✗ Cannot list 448 directory contents"
                else
                    echo "  ✗ 448 directory does not exist"
                fi
            } > "$error_file"
            
            return 2
        fi
        
        echo "[BATCH-$batch_id] Found ${#camera_files[@]} camera files for $take_name: ${camera_files[*]}"
        
        local best_result_dir="${BEST_OUT_DIR}/${take_name}"
        local need_processing=false
        
        if [[ ! -d "$best_result_dir" ]] || [[ ! -f "${best_result_dir}/metadata.json" ]]; then
            need_processing=true
            echo "[BATCH-$batch_id] [CHECK] Best result not found for $take_name, checking for existing ViPE results..."
            
            local all_cameras_have_results=true
            for camera in "${camera_files[@]}"; do
                if ! check_vipe_results_exist "$take_name" "$camera"; then
                    all_cameras_have_results=false
                    break
                fi
            done
            
            if [[ "$all_cameras_have_results" == true ]]; then
                echo "[BATCH-$batch_id] [SKIP] All ViPE results exist for $take_name, skipping inference and starting from rendering..."
            else
                echo "[BATCH-$batch_id] [INFER] Some ViPE results missing, will run full pipeline..."
            fi
        else
            echo "[BATCH-$batch_id] [SKIP] Best result already exists for $take_name: $best_result_dir"
            echo "[BATCH-$batch_id] ✅ Skipping take: $take_name"
            return 0
        fi
        
        declare -A camera_result_subdirs
        
        for camera in "${camera_files[@]}"; do
            echo "[BATCH-$batch_id] [START] ${take_name}/${camera}"
            
            local skip_inference=false
            if check_vipe_results_exist "$take_name" "$camera"; then
                skip_inference=true
                echo "[BATCH-$batch_id] [SKIP] ViPE results already exist for ${take_name}/${camera}, skipping inference"
            fi
            
            if [[ "$skip_inference" == false ]]; then
                mkdir -p "${VIPE_RESULTS_ROOT}/${take_name}"
                run_inference "$take_name" "$camera" "lyra" "$uuid"
            else
                local found_source=""
                local possible_sources=(
                    "${DEFAULT_VIPE_ROOT}/${take_name}"
                    "$(pwd)/vipe_results/${take_name}"
        "${WORKING_DIR}/vipe_results/${take_name}"
        "${DATA_DIR}/vipe_results/${take_name}"
                )
                
                for source in "${possible_sources[@]}"; do
                    if [[ -d "$source" ]]; then
                        found_source="$source"
                        mkdir -p "${VIPE_RESULTS_ROOT}/${take_name}"
                        rsync -a "${source}/" "${VIPE_RESULTS_ROOT}/${take_name}/"
                        echo "[BATCH-$batch_id] [SYNC] Found and synced results from: $source"
                        break
                    fi
                done
            fi
            
            latest_subdir=$(find_latest_result_subdir "$take_name" "$camera")
            camera_result_subdirs["$camera"]="$latest_subdir"
            
            run_render "$take_name" "$latest_subdir" "$uuid"
            
            # render_vipe_pointcloud.py uses input_dir stem as video_name, saves to ${OUTPUT_DIR}/${video_name}
            # input_dir = ${VIPE_RESULTS_ROOT}/${take_name}/${result_subdir}
            # output path = ${OUTPUT_DIR}/${result_subdir}
            if [[ -n "$latest_subdir" ]]; then
                render_dir_base="$OUTPUT_DIR/${latest_subdir}"
                if [[ -d "$render_dir_base" ]]; then
                    video_path="$render_dir_base/ego_Prior.mp4"
                    if [[ -f "$video_path" ]]; then
                        if [[ ! -f "$render_dir_base/frame_000000.png" ]]; then
                            echo "[BATCH-$batch_id] [POST] Extracting frames from $video_path"
                            if command -v ffmpeg >/dev/null 2>&1; then
                                ffmpeg -hide_banner -loglevel error -i "$video_path" -start_number 0 "$render_dir_base/frame_%06d.png" 2>/dev/null || {
                                    echo "[BATCH-$batch_id] [POST] WARNING: Failed to extract frames from MP4"
                                }
                            else
                                echo "[BATCH-$batch_id] [POST] WARNING: ffmpeg not found, cannot extract frames"
                            fi
                        else
                            echo "[BATCH-$batch_id] [POST] Frames already extracted, skipping"
                        fi
                    fi
                    
                    echo "[BATCH-$batch_id] [POST] Post-processing render dir: $render_dir_base"
                    post_out=$(python "${WORKING_DIR}/scripts/postprocess_render_dir.py" --dir "$render_dir_base" --threshold 30.0 2>&1 | tr -d '\r')
                    echo "[BATCH-$batch_id] [POST] Raw output: $post_out"
                    
                    total_white=$(echo "$post_out" | awk -F'=' '/total_white_pixels/ {print $2}')
                    frames_nonblack=$(echo "$post_out" | awk -F'=' '/frames_with_any_white/ {print $2}')
                    
                    if [[ -z "$total_white" ]] || [[ "$total_white" == "" ]]; then
                        total_white="0"
                    fi
                    if [[ -z "$frames_nonblack" ]] || [[ "$frames_nonblack" == "" ]]; then
                        frames_nonblack="0"
                    fi
                    
                    eval "metric_total_white_${camera}=$total_white"
                    eval "metric_frames_nonblack_${camera}=$frames_nonblack"
                    echo "[BATCH-$batch_id] [POST] Camera $camera metrics: frames_nonblack=$frames_nonblack, total_white=$total_white"
                else
                    echo "[BATCH-$batch_id] [POST] Render dir not found: $render_dir_base"
                    echo "[BATCH-$batch_id] [POST] Available directories in $OUTPUT_DIR:"
                    ls -1d "$OUTPUT_DIR"/* 2>/dev/null | head -5 || echo "  (none)"
                fi
            else
                echo "[BATCH-$batch_id] [POST] No result subdir found for ${take_name}/${camera}"
            fi

            if [[ -n "$latest_subdir" ]]; then
                echo "[BATCH-$batch_id] [SAVE] Keeping ViPE results for ${take_name}/${latest_subdir}"
                if [[ -d "${VIPE_RESULTS_ROOT}/${take_name}/${latest_subdir}" ]]; then
                    echo "[BATCH-$batch_id] [SAVE] ViPE results saved in workspace: ${VIPE_RESULTS_ROOT}/${take_name}/${latest_subdir}"
                fi
                if [[ -d "${DEFAULT_VIPE_ROOT}/${take_name}/${latest_subdir}" ]]; then
                    echo "[BATCH-$batch_id] [SAVE] ViPE results also available in default location: ${DEFAULT_VIPE_ROOT}/${take_name}/${latest_subdir}"
                fi
            fi
        done

        # Select best camera: priority 1) frames_with_any_white, 2) total_white_pixels
        best_camera=""
        best_frames=-1
        best_total=-1
        echo "[BATCH-$batch_id] [BEST] Selecting best camera from ${#camera_files[@]} cameras..."
        for camera in "${camera_files[@]}"; do
            frames_val=$(eval echo \${metric_frames_nonblack_${camera}:-"-1"})
            total_val=$(eval echo \${metric_total_white_${camera}:-"-1"})
            if [[ -z "$frames_val" ]] || [[ "$frames_val" == "" ]]; then frames_val=-1; fi
            if [[ -z "$total_val" ]] || [[ "$total_val" == "" ]]; then total_val=-1; fi
            
            echo "[BATCH-$batch_id] [BEST] Camera $camera: frames=$frames_val, total=$total_val"
            
            if [[ "$frames_val" != "-1" ]] || [[ "$total_val" != "-1" ]]; then
                if (( frames_val > best_frames )) || { (( frames_val == best_frames )) && (( total_val > best_total )); }; then
                    best_frames=$frames_val
                    best_total=$total_val
                    best_camera=$camera
                    echo "[BATCH-$batch_id] [BEST] New best: $camera (frames=$frames_val, total=$total_val)"
                fi
            fi
        done

        if [[ -n "$best_camera" ]]; then
            echo "[BATCH-$batch_id] [BEST] Selected best camera: $best_camera (frames=$best_frames, total=$best_total)"
            best_result_subdir="${camera_result_subdirs[$best_camera]}"
            
            if [[ -z "$best_result_subdir" ]]; then
                echo "[BATCH-$batch_id] [BEST] WARNING: No result_subdir found for best camera $best_camera, trying to find it..."
                best_result_subdir=$(find_latest_result_subdir "$take_name" "$best_camera")
            fi
            
            best_render_dir="$OUTPUT_DIR/${best_result_subdir}"
            
            if [[ -d "$best_render_dir" ]]; then
                dest_dir="$BEST_OUT_DIR/${take_name}"
                ego_prior_dir="$dest_dir/ego_Prior"
                exo_gt_dir="$dest_dir/exo_GT"
                ego_gt_dir="$dest_dir/ego_GT"
                mkdir -p "$ego_prior_dir" "$exo_gt_dir" "$ego_gt_dir"
                echo "[BATCH-$batch_id] [BEST] Copying best render ($best_camera, subdir: $best_result_subdir) to $ego_prior_dir"
                
                if [[ -f "$best_render_dir/ego_Prior.mp4" ]]; then
                    cp "$best_render_dir/ego_Prior.mp4" "$ego_prior_dir/"
                else
                    rsync -a --delete "$best_render_dir/" "$ego_prior_dir/"
                fi
                
                meta_path="$dest_dir/metadata.json"
                render_end=$((END_FRAME - START_FRAME))
                src_video_path="${BASE_PATH}/${take_name}/frame_aligned_videos/downscaled/448/${best_camera}.mp4"
                {
                    echo '{'
                    echo "  \"take_name\": \"$take_name\"," 
                    echo "  \"ego_uuid\": \"$uuid\"," 
                    echo "  \"best_camera\": \"$best_camera\"," 
                    echo "  \"source_video\": \"$src_video_path\"," 
                    echo "  \"render_frame_start\": 0," 
                    echo "  \"render_frame_end\": $render_end," 
                    echo "  \"source_frame_start\": $START_FRAME," 
                    echo "  \"source_frame_end\": $END_FRAME," 
                    echo "  \"metrics\": {\"frames_with_any_white\": $best_frames, \"total_white_pixels\": $best_total},"
                    echo "  \"frames\": ["
                    for ((i=0; i<=render_end; i++)); do
                        src_idx=$((START_FRAME + i))
                        printf "    {\"render_index\": %d, \"source_frame\": %d, \"filename\": \"frame_%06d.png\"}" "$i" "$src_idx" "$i"
                        if [[ $i -lt $render_end ]]; then
                            echo ","
                        else
                            echo
                        fi
                    done
                    echo "  ]"
                    echo '}'
                } > "$meta_path"

                find "$ego_prior_dir" -type f -name "*_mask.png" -delete

                frames_to_extract=49
                if command -v ffmpeg >/dev/null 2>&1; then
                    echo "[BATCH-$batch_id] [GT] Extracting EXO GT frames from $src_video_path -> $exo_gt_dir (start=$START_FRAME, count=$frames_to_extract)"
                    exo_end_frame_abs=$((START_FRAME + frames_to_extract))
                    ffmpeg -hide_banner -loglevel error -y -i "$src_video_path" -vf "trim=start_frame=$START_FRAME:end_frame=$exo_end_frame_abs,setpts=PTS-STARTPTS" -start_number 0 -vsync 0 "$exo_gt_dir/frame_%06d.png"
                else
                    echo "[BATCH-$batch_id] [GT][WARN] ffmpeg not found; skipping EXO GT extraction"
                fi
                
                ego_video_path="${BASE_PATH}/${take_name}/frame_aligned_videos/downscaled/448/aria01_214-1.mp4"
                if [[ -f "$ego_video_path" ]]; then
                    if command -v ffmpeg >/dev/null 2>&1; then
                        echo "[BATCH-$batch_id] [GT] Extracting EGO GT frames from $ego_video_path -> $ego_gt_dir (start=$START_FRAME, count=$frames_to_extract)"
                        ego_end_frame_abs=$((START_FRAME + frames_to_extract))
                        ffmpeg -hide_banner -loglevel error -y -i "$ego_video_path" -vf "trim=start_frame=$START_FRAME:end_frame=$ego_end_frame_abs,setpts=PTS-STARTPTS" -start_number 0 -vsync 0 "$ego_gt_dir/frame_%06d.png"
                    else
                        echo "[BATCH-$batch_id] [GT][WARN] ffmpeg not found; skipping EGO GT extraction"
                    fi
                else
                    echo "[BATCH-$batch_id] [GT][WARN] Ego GT video not found: $ego_video_path"
                fi

                if [[ -d "$OUTPUT_DIR/${take_name}" ]]; then
                    echo "[BATCH-$batch_id] [CLEAN] Removing full ego_view_rendering for take: $take_name"
                    rm -rf "$OUTPUT_DIR/${take_name}"
                fi
            else
                echo "[BATCH-$batch_id] [BEST] Best render directory not found for $take_name"
            fi
        else
            echo "[BATCH-$batch_id] [BEST] No best camera determined for $take_name"
        fi

        if [[ -d "${VIPE_RESULTS_ROOT}/${take_name}" ]]; then
            echo "[BATCH-$batch_id] [SAVE] Keeping workspace vipe_results take dir: ${VIPE_RESULTS_ROOT}/${take_name}"
        fi
        if [[ -d "${DEFAULT_VIPE_ROOT}/${take_name}" ]]; then
            echo "[BATCH-$batch_id] [SAVE] Keeping default vipe_results take dir: ${DEFAULT_VIPE_ROOT}/${take_name}"
        fi
        
        echo "[BATCH-$batch_id] ✅ Completed take: $take_name"
        
    } > "$log_file" 2>&1
    
    cat "$log_file"
}

create_or_load_uuid_mapping() {
    local mapping_file="$1"
    local ego_pose_dir="$2"
    
    if [[ -f "$mapping_file" ]]; then
        echo "📖 Using existing UUID mapping file: $mapping_file" >&2
        return 0
    fi
    
    echo "🔨 Creating UUID mapping from ego_pose JSON files..." >&2
    
    python3 << PYTHON_SCRIPT
import json
import os
import sys

try:
    ego_pose_dir = "$ego_pose_dir"
    mapping_file = "$mapping_file"
    
    if not os.path.isdir(ego_pose_dir):
        print(f"ERROR: Ego pose directory not found: {ego_pose_dir}", file=sys.stderr)
        sys.exit(1)
    
    uuid_mapping = {}
    
    json_files = [f for f in os.listdir(ego_pose_dir) if f.endswith('.json')]
    print(f"Processing {len(json_files)} ego pose files...", file=sys.stderr)
    
    for json_file in json_files:
        filepath = os.path.join(ego_pose_dir, json_file)
        uuid = json_file.replace('.json', '')
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            metadata = data.get('metadata', {})
            take_name = metadata.get('take_name')
            take_uid = metadata.get('take_uid')
            
            if take_name:
                if take_name in uuid_mapping and uuid_mapping[take_name] != uuid:
                    print(f"WARNING: Duplicate take_name '{take_name}' with different UUIDs: {uuid_mapping[take_name]} vs {uuid}", file=sys.stderr)
                uuid_mapping[take_name] = uuid
        except Exception as e:
            print(f"WARNING: Failed to process {json_file}: {e}", file=sys.stderr)
            continue
    
    os.makedirs(os.path.dirname(mapping_file), exist_ok=True)
    with open(mapping_file, 'w') as f:
        json.dump(uuid_mapping, f, indent=2)
    
    print(f"✅ Created UUID mapping file with {len(uuid_mapping)} entries: {mapping_file}", file=sys.stderr)
    
except Exception as e:
    print(f"ERROR: Failed to create UUID mapping: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT
}

load_uuid_from_mapping() {
    local mapping_file="$1"
    
    if [[ ! -f "$mapping_file" ]]; then
        echo "ERROR: UUID mapping file not found: $mapping_file" >&2
        return 1
    fi
    
    python3 << PYTHON_SCRIPT
import json
import sys

try:
    mapping_file = "$mapping_file"
    with open(mapping_file, 'r') as f:
        uuid_mapping = json.load(f)
    
    for take_name, uuid in uuid_mapping.items():
        print(f"{take_name}={uuid}")
except Exception as e:
    print(f"ERROR: Failed to load UUID mapping: {e}", file=sys.stderr)
    sys.exit(1)
PYTHON_SCRIPT
}

get_camera_files() {
    local take_dir="$1"
    local video_dir="${take_dir}/frame_aligned_videos/downscaled/448"
    
    if [[ ! -d "$video_dir" ]]; then
        echo "WARNING: Video directory not found: $video_dir"
        return 1
    fi
    
    # cam으로 시작하는 mp4 파일들을 찾아서 확장자 제거
    find "$video_dir" -name "cam*.mp4" -type f | sed 's/.*\///' | sed 's/\.mp4$//' | sort
}

check_vipe_results_exist() {
    local take_name="$1"
    local camera="$2"
    
    local search_paths=(
        "${VIPE_RESULTS_ROOT}/${take_name}"
        "${DEFAULT_VIPE_ROOT}/${take_name}"
        "$(pwd)/vipe_results/${take_name}"
        "$(pwd)/workspace/vipe_results/${take_name}"
        "${WORKING_DIR}/vipe_results/${take_name}"
        "${DATA_DIR}/vipe_results/${take_name}"
    )
    
    for path in "${search_paths[@]}"; do
        if [[ -n "$camera" ]]; then
            if ls -1d "$path"/${camera}_* 2>/dev/null | head -n 1 | grep -q .; then
                return 0
            fi
        else
            if [[ -d "$path" ]]; then
                return 0
            fi
        fi
    done
    
    return 1
}

find_latest_result_subdir() {
    local take_name="$1"
    local camera="$2"
    
    local search_paths=(
        "${VIPE_RESULTS_ROOT}/${take_name}"
        "${DEFAULT_VIPE_ROOT}/${take_name}"
        "${WORKING_DIR}/vipe_results/${take_name}"  # working dir의 vipe_results
        "${DATA_DIR}/vipe_results/${take_name}"  # data dir의 vipe_results
    )
    
    local results_root=""
    for path in "${search_paths[@]}"; do
        if [[ -d "$path" ]]; then
            results_root="$path"
            break
        fi
    done
    
    if [[ -z "$results_root" ]]; then
        echo ""
        return 0
    fi
    
    if [[ -n "$camera" ]]; then
        local latest
        latest=$(ls -1dt "$results_root"/${camera}_* 2>/dev/null | head -n 1)
        if [[ -n "$latest" ]]; then
            basename "$latest"
            return 0
        fi
    fi
    
    local latest
    latest=$(ls -1t "$results_root" 2>/dev/null | head -n 1)
    if [[ -n "$latest" ]]; then
        basename "$latest"
    else
        echo ""
    fi
}

create_meta_json() {
    local ego_pose_path="$1"
    local take_name="$2"
    local result_subdir="$3"
    local output_path="$4"
    local start_frame="${5:-$START_FRAME}"
    local end_frame="${6:-$END_FRAME}"
    
    if [[ ! -f "$ego_pose_path" ]]; then
        echo "ERROR: Ego camera pose not found: $ego_pose_path"
        return 1
    fi
    
    local camera_name="cam03"
    if [[ "$result_subdir" =~ ^(cam[0-9]+) ]]; then
        camera_name="${BASH_REMATCH[1]}"
    fi
    
    python3 << PYTHON_SCRIPT
import json
import sys
from pathlib import Path

try:
    ego_pose_path = "$ego_pose_path"
    take_name = "$take_name"
    result_subdir = "$result_subdir"
    output_path = "$output_path"
    start_frame = int("$start_frame")
    end_frame = int("$end_frame")
    camera_name = "$camera_name"
    
    with open(ego_pose_path, 'r') as f:
        ego_data = json.load(f)
    
    exo_intrinsics = None
    exo_extrinsics = None
    
    if camera_name in ego_data:
        cam_data = ego_data[camera_name]
        if 'camera_intrinsics' in cam_data:
            exo_intrinsics = cam_data['camera_intrinsics']
        if 'camera_extrinsics' in cam_data:
            exo_extrinsics = cam_data['camera_extrinsics']
    else:
        available_cams = [k for k in ego_data.keys() if k.startswith('cam')]
        print(f"ERROR: Camera '{camera_name}' not found in ego_pose. Available: {available_cams}", file=sys.stderr)
        sys.exit(1)
    
    ego_intrinsics = None
    ego_extrinsics_list = []
    
    ego_camera_key = None
    for key in ego_data.keys():
        if key.startswith('aria'):
            ego_camera_key = key
            break
    
    if ego_camera_key and ego_camera_key in ego_data:
        aria_data = ego_data[ego_camera_key]
        if 'camera_intrinsics' in aria_data:
            ego_intrinsics = aria_data['camera_intrinsics']
        
        if 'camera_extrinsics' in aria_data:
            ego_ext = aria_data['camera_extrinsics']
            
            if isinstance(ego_ext, dict):
                frame_keys = sorted([int(k) for k in ego_ext.keys() if str(k).isdigit()])
                frames_in_range = [f for f in frame_keys if start_frame <= f <= end_frame]
                
                if not frames_in_range:
                    print(f"ERROR: No frames found in range {start_frame}-{end_frame}. Available: {frame_keys[0] if frame_keys else 'N/A'} ~ {frame_keys[-1] if frame_keys else 'N/A'}", file=sys.stderr)
                    sys.exit(1)
                
                ego_extrinsics_list = [ego_ext[str(f)] for f in sorted(frames_in_range)]
            elif isinstance(ego_ext, list):
                if len(ego_ext) > 0 and isinstance(ego_ext[0], list):
                    if len(ego_ext[0]) == 4:
                        if start_frame == 0 and end_frame >= len(ego_ext) - 1:
                            ego_extrinsics_list = ego_ext
                        else:
                            ego_extrinsics_list = ego_ext[start_frame:end_frame+1]
                    elif len(ego_ext) == 3 and len(ego_ext[0]) == 4:
                        ego_extrinsics_list = [ego_ext]
                    else:
                        ego_extrinsics_list = [ego_ext] if len(ego_ext) == 3 else []
                else:
                    ego_extrinsics_list = [ego_ext] if len(ego_ext) == 3 else []
            else:
                ego_extrinsics_list = []
    else:
        print(f"ERROR: No aria camera found in ego_pose file", file=sys.stderr)
        sys.exit(1)
    
    if exo_intrinsics is None:
        print(f"ERROR: exo_intrinsics not found for camera {camera_name}", file=sys.stderr)
        sys.exit(1)
    
    if exo_extrinsics is None:
        print(f"ERROR: exo_extrinsics not found for camera {camera_name}", file=sys.stderr)
        sys.exit(1)
    
    if ego_intrinsics is None:
        print(f"ERROR: ego_intrinsics not found", file=sys.stderr)
        sys.exit(1)
    
    if not ego_extrinsics_list:
        print(f"ERROR: No ego_extrinsics found for frames {start_frame}-{end_frame}", file=sys.stderr)
        sys.exit(1)
    
    # render_vipe_pointcloud.py uses input_dir stem as video_name, matches with exo_path parent
    # result_subdir becomes video_name (e.g., cam04_video_cmu_bike08_2_cam04_2582104)
    exo_path = f"./takes/{result_subdir}/cam01.mp4"
    
    meta_data = {
        "test_datasets": [{
            "exo_path": exo_path,
            "camera_intrinsics": exo_intrinsics,
            "camera_extrinsics": exo_extrinsics,
            "ego_intrinsics": ego_intrinsics,
            "ego_extrinsics": ego_extrinsics_list
        }]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    print(f"✅ Created meta.json: {output_path}", file=sys.stderr)
    print(f"   exo_path: {exo_path}", file=sys.stderr)
    print(f"   video_name (result_subdir): {result_subdir}", file=sys.stderr)
    print(f"   exo camera: {camera_name}", file=sys.stderr)
    print(f"   frame range: {start_frame}-{end_frame} ({len(ego_extrinsics_list)} frames)", file=sys.stderr)
    print(f"   exo_intrinsics: {exo_intrinsics is not None}", file=sys.stderr)
    print(f"   exo_extrinsics: {exo_extrinsics is not None}", file=sys.stderr)
    print(f"   ego_intrinsics: {ego_intrinsics is not None}", file=sys.stderr)
    print(f"   ego_extrinsics count: {len(ego_extrinsics_list)}", file=sys.stderr)
    
except Exception as e:
    print(f"ERROR: Failed to create meta.json: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYTHON_SCRIPT
}

run_render() {
    local take_name=$1
    local result_subdir=$2
    local ego_uuid=$3
    
    if [[ -z "$result_subdir" ]]; then
        echo "WARNING: No ViPE result found for ${take_name}, skipping render"
        return 1
    fi
    
    local input_dir="${VIPE_RESULTS_ROOT}/${take_name}/${result_subdir}"
    local ego_camera_pose_path="${DATA_DIR}/annotations/ego_pose/test/camera_pose/${ego_uuid}.json"
    local online_calibration_path="${DATA_DIR}/takes/${take_name}/trajectory/online_calibration.jsonl"
    
    local meta_json_path="${META_FILES_DIR}/meta_${take_name}_${result_subdir}.json"
    create_meta_json "$ego_camera_pose_path" "$take_name" "$result_subdir" "$meta_json_path" "$START_FRAME" "$END_FRAME"
    
    if [[ ! -f "$meta_json_path" ]]; then
        echo "ERROR: Failed to create meta.json: $meta_json_path"
        return 1
    fi
    
    echo "[RENDER] =========================================="
    echo "[RENDER] Rendering: ${take_name} | result: ${result_subdir}"
    echo "[RENDER] InputDir: $input_dir"
    echo "[RENDER] MetaJson: $meta_json_path"
    echo "[RENDER] EgoPose: $ego_camera_pose_path"
    echo "[RENDER] OnlineCalib: $online_calibration_path"
    echo "[RENDER] Frames: $START_FRAME-$END_FRAME"
    echo "[RENDER] =========================================="
    
    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: Input result directory not found: $input_dir"
        
        local error_file="$ERROR_ROOT/${take_name}_${result_subdir}_render_input.error"
        {
            echo "ERROR: Input result directory not found for rendering"
            echo "Timestamp: $(date)"
            echo "Take name: $take_name"
            echo "Result subdir: $result_subdir"
            echo "Expected input directory: $input_dir"
            echo "UUID: $ego_uuid"
            echo ""
            echo "Directory structure check:"
            if [[ -d "${VIPE_RESULTS_ROOT}/${take_name}" ]]; then
                echo "  ✓ ViPE results root exists: ${VIPE_RESULTS_ROOT}/${take_name}"
                ls -la "${VIPE_RESULTS_ROOT}/${take_name}" 2>/dev/null || echo "  ✗ Cannot list ViPE results contents"
            else
                echo "  ✗ ViPE results root does not exist: ${VIPE_RESULTS_ROOT}/${take_name}"
            fi
            echo ""
            if [[ -d "${DEFAULT_VIPE_ROOT}/${take_name}" ]]; then
                echo "  ✓ Default ViPE results root exists: ${DEFAULT_VIPE_ROOT}/${take_name}"
                ls -la "${DEFAULT_VIPE_ROOT}/${take_name}" 2>/dev/null || echo "  ✗ Cannot list default ViPE results contents"
            else
                echo "  ✗ Default ViPE results root does not exist: ${DEFAULT_VIPE_ROOT}/${take_name}"
            fi
        } > "$error_file"
        
        return 1
    fi
    if [[ ! -f "$ego_camera_pose_path" ]]; then
        echo "ERROR: Ego camera pose not found: $ego_camera_pose_path"
        
        local error_file="$ERROR_ROOT/${take_name}_${result_subdir}_render_ego_pose.error"
        {
            echo "ERROR: Ego camera pose file not found for rendering"
            echo "Timestamp: $(date)"
            echo "Take name: $take_name"
            echo "Result subdir: $result_subdir"
            echo "Expected ego pose path: $ego_camera_pose_path"
            echo "UUID: $ego_uuid"
            echo ""
            echo "File existence check:"
            if [[ -f "$ego_camera_pose_path" ]]; then
                echo "  ✓ Ego pose file exists"
            else
                echo "  ✗ Ego pose file does not exist"
                echo "  Directory contents:"
                ls -la "$(dirname "$ego_camera_pose_path")" 2>/dev/null || echo "  ✗ Cannot list directory contents"
            fi
        } > "$error_file"
        
        return 1
    fi
    if [[ ! -f "$online_calibration_path" ]]; then
        echo "ERROR: Online calibration not found: $online_calibration_path"
        
        local error_file="$ERROR_ROOT/${take_name}_${result_subdir}_render_calibration.error"
        {
            echo "ERROR: Online calibration file not found for rendering"
            echo "Timestamp: $(date)"
            echo "Take name: $take_name"
            echo "Result subdir: $result_subdir"
            echo "Expected calibration path: $online_calibration_path"
            echo "UUID: $ego_uuid"
            echo ""
            echo "File existence check:"
            if [[ -f "$online_calibration_path" ]]; then
                echo "  ✓ Calibration file exists"
            else
                echo "  ✗ Calibration file does not exist"
                echo "  Directory contents:"
                ls -la "$(dirname "$online_calibration_path")" 2>/dev/null || echo "  ✗ Cannot list directory contents"
            fi
        } > "$error_file"
        
        return 1
    fi
    
    local render_start=0
    local render_end=$((END_FRAME - START_FRAME))
    if [[ $render_end -lt 0 ]]; then
        render_end=0
    fi
    
    if [[ -z "${REPO_ROOT:-}" ]]; then
        echo "ERROR: REPO_ROOT is not set. Please configure it in config.sh"
        return 1
    fi
    
    if [[ -z "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    else
        export PYTHONPATH="${REPO_ROOT}:$PYTHONPATH"
    fi
    
    cd "${REPO_ROOT}" || {
        echo "ERROR: Cannot change to REPO_ROOT: ${REPO_ROOT}"
        return 1
    }
    python scripts/render_vipe_pointcloud.py \
        --input_dir "$input_dir" \
        --out_dir "$OUTPUT_DIR" \
        --meta_json_path "$meta_json_path" \
        --point_size "$POINT_SIZE" \
        --start_frame "$render_start" \
        --end_frame "$render_end" \
        --fish_eye_rendering \
        --use_mean_bg \
        --only_bg \
        --online_calibration_path "$online_calibration_path"
}

run_inference() {
    local take_name=$1
    local camera=$2
    local pipeline=$3
    local uuid=$4
    
    local video_path="${BASE_PATH}/${take_name}/frame_aligned_videos/downscaled/448/${camera}.mp4"
    
    echo "=========================================="
    echo "Processing: ${take_name}/${camera}"
    echo "Video path: $video_path"
    echo "Frame range: $START_FRAME to $END_FRAME"
    echo "Pipeline: $pipeline"
    echo "=========================================="
    
    if [[ ! -f "$video_path" ]]; then
        echo "ERROR: Video file not found: $video_path"
        
        local error_file="$ERROR_ROOT/${take_name}_${camera}_inference.error"
        {
            echo "ERROR: Video file not found for inference"
            echo "Timestamp: $(date)"
            echo "Take name: $take_name"
            echo "Camera: $camera"
            echo "Pipeline: $pipeline"
            echo "Expected video path: $video_path"
            echo "Frame range: $START_FRAME to $END_FRAME"
            echo ""
            echo "Directory structure check:"
            local base_dir="${BASE_PATH}/${take_name}"
            if [[ -d "$base_dir" ]]; then
                echo "  ✓ Take base directory exists: $base_dir"
                ls -la "$base_dir" 2>/dev/null || echo "  ✗ Cannot list take base directory contents"
            else
                echo "  ✗ Take base directory does not exist: $base_dir"
            fi
            echo ""
            local video_dir="${base_dir}/frame_aligned_videos/downscaled/448"
            if [[ -d "$video_dir" ]]; then
                echo "  ✓ Video directory exists: $video_dir"
                echo "  Files in video directory:"
                ls -la "$video_dir" 2>/dev/null || echo "  ✗ Cannot list video directory contents"
            else
                echo "  ✗ Video directory does not exist: $video_dir"
            fi
        } > "$error_file"
        
        return 1
    fi
    
    if [[ -z "${EGO4D_DATASET_TYPE:-}" ]]; then
        echo "ERROR: EGO4D_DATASET_TYPE is not set. Please configure it in config.sh"
        return 1
    fi
    export EGO4D_DATASET_TYPE="${EGO4D_DATASET_TYPE}"
    
    if [[ -z "${REPO_ROOT:-}" ]]; then
        echo "ERROR: REPO_ROOT is not set. Please configure it in config.sh"
        return 1
    fi
    
    if [[ -z "${PYTHONPATH:-}" ]]; then
        export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
    else
        export PYTHONPATH="${REPO_ROOT}:$PYTHONPATH"
    fi

    local total_frames=$(ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 "$video_path" 2>/dev/null || echo "0")
    if [[ -z "$total_frames" ]] || [[ "$total_frames" == "0" ]]; then
        echo "ERROR: Cannot determine video frame count for: $video_path"
        return 1
    fi
    
    local actual_start=$START_FRAME
    local actual_end=$END_FRAME
    if [[ $actual_start -ge $total_frames ]]; then
        echo "ERROR: Start frame ($actual_start) >= total frames ($total_frames) for: $video_path"
        return 1
    fi
    if [[ $actual_end -ge $total_frames ]]; then
        echo "WARNING: End frame ($actual_end) >= total frames ($total_frames), adjusting to $((total_frames - 1))"
        actual_end=$((total_frames - 1))
    fi
    
    mkdir -p "${VIPE_RESULTS_ROOT}/${take_name}"
    
    # Convert video_path to absolute path
    local abs_video_path
    if [[ "$video_path" = /* ]]; then
        abs_video_path="$video_path"
    else
        abs_video_path="$(cd "$(dirname "$video_path")" && pwd)/$(basename "$video_path")"
    fi
    
    # Verify absolute path exists
    if [[ ! -f "$abs_video_path" ]]; then
        echo "ERROR: Video file not found at absolute path: $abs_video_path"
        return 1
    fi
    
    local video_basename=$(basename "$abs_video_path" .mp4)
    local temp_video_dir="${TEMP_DIR}/video_${take_name}_${camera}_$$"
    mkdir -p "$temp_video_dir"
    
    # Create symbolic link with absolute path
    ln -sf "$abs_video_path" "$temp_video_dir/${camera}.mp4"
    local temp_video_path="$temp_video_dir/${camera}.mp4"
    
    # Verify symbolic link was created
    if [[ ! -f "$temp_video_path" ]] && [[ ! -L "$temp_video_path" ]]; then
        echo "ERROR: Failed to create symbolic link: $temp_video_path -> $abs_video_path"
        return 1
    fi
    
    cd "${REPO_ROOT}" || {
        echo "ERROR: Cannot change to REPO_ROOT: ${REPO_ROOT}"
        return 1
    }
    
    # Extract GT intrinsics from ego_pose if USE_GT_INTRINSIC=true
    local gt_intrinsic_arg=""
    if [[ -n "${USE_GT_INTRINSIC:-}" ]] && [[ "$USE_GT_INTRINSIC" == "true" ]]; then
        local ego_pose_file="${DATA_DIR}/annotations/ego_pose/test/camera_pose/${uuid}.json"
        if [[ -f "$ego_pose_file" ]]; then
            gt_intrinsic_arg=$(python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
cam = sys.argv[2]
if cam in d and 'camera_intrinsics' in d[cam]:
    print(json.dumps(d[cam]['camera_intrinsics']))
" "$ego_pose_file" "$camera" 2>/dev/null)
            if [[ -n "$gt_intrinsic_arg" ]]; then
                echo "Using GT exo intrinsic for ${camera}: ${gt_intrinsic_arg}"
            fi
        fi
    fi

    if [[ -n "$gt_intrinsic_arg" ]]; then
        vipe infer "$temp_video_path" \
            --start_frame $actual_start \
            --end_frame $actual_end \
            --assume_fixed_camera_pose \
            --pipeline $pipeline \
            --use_exo_intrinsic_gt "$gt_intrinsic_arg" \
            --output "${VIPE_RESULTS_ROOT}/${take_name}"
    else
        vipe infer "$temp_video_path" \
            --start_frame $actual_start \
            --end_frame $actual_end \
            --assume_fixed_camera_pose \
            --pipeline $pipeline \
            --output "${VIPE_RESULTS_ROOT}/${take_name}"
    fi
    
    if [[ -d "${VIPE_RESULTS_ROOT}/${take_name}/${camera}" ]]; then
        :
    elif [[ -d "${VIPE_RESULTS_ROOT}/${take_name}" ]] && [[ -n "$(ls -A "${VIPE_RESULTS_ROOT}/${take_name}" 2>/dev/null)" ]]; then
        local result_subdir=$(ls -1t "${VIPE_RESULTS_ROOT}/${take_name}" 2>/dev/null | head -1)
        if [[ -n "$result_subdir" ]] && [[ "$result_subdir" != "$camera" ]]; then
            mv "${VIPE_RESULTS_ROOT}/${take_name}/${result_subdir}" "${VIPE_RESULTS_ROOT}/${take_name}/${camera}_${result_subdir}" 2>/dev/null || true
        fi
    fi
    
    rm -rf "$temp_video_dir" 2>/dev/null || true
        
    echo "Completed: ${take_name}/${camera}"
    echo ""
}

main() {
    parse_arguments "$@"
    
    create_or_load_uuid_mapping "$UUID_MAPPING_FILE" "$EGO_POSE_DIR"
    
    echo "Loading UUID mapping from: $UUID_MAPPING_FILE"
    declare -A uuid_mapping
    while IFS='=' read -r key value; do
        if [[ -n "$key" ]] && [[ -n "$value" ]]; then
            uuid_mapping["$key"]="$value"
        fi
    done < <(load_uuid_from_mapping "$UUID_MAPPING_FILE")

    if [[ ${#uuid_mapping[@]} -eq 0 ]]; then
        echo "❌ ERROR: No UUID mappings found in mapping file"
        return 1
    fi

    echo "Loaded ${#uuid_mapping[@]} UUID mappings from mapping file"
    
    echo "📁 Data storage path: $DATA_ROOT"
    echo "   Frame range: $START_FRAME ~ $END_FRAME"
    echo "   Rendering output: $OUTPUT_DIR"
    echo "   Best results: $BEST_OUT_DIR"
    echo "   ViPE results: $VIPE_RESULTS_ROOT"
    echo "   Error logs: $ERROR_ROOT"
    echo ""
    
    echo "🚀 Processing with $BATCH_SIZE parallel processes."
    echo ""
    
    cleanup_zombies &
    local zombie_cleaner_pid=$!
    
    local takes_to_process=()
    local total_takes_found=0
    local takes_with_uuid=0
    
    if [[ ! -d "$TAKES_DIR" ]]; then
        echo "❌ ERROR: Takes directory does not exist: $TAKES_DIR"
        return 1
    fi
    
    echo "🔍 Loading all takes from: $TAKES_DIR"
    local all_takes_list=()
    while IFS= read -r take_name || [[ -n "$take_name" ]]; do
        if [[ -n "$take_name" ]]; then
            all_takes_list+=("$take_name")
        fi
    done < <(load_all_takes "$TAKES_DIR")
    
    if [[ ${#all_takes_list[@]} -eq 0 ]]; then
        echo "❌ ERROR: No takes found in directory: $TAKES_DIR"
        return 1
    fi
    
    echo "   Loaded ${#all_takes_list[@]} takes from directory"
    echo ""
    
    echo "🔍 Processing all takes..."
    for take_name in "${all_takes_list[@]}"; do
        take_dir="${TAKES_DIR}/${take_name}"
        
        if [[ ! -d "$take_dir" ]]; then
            echo "WARNING: Take directory not found: $take_dir (skipping)"
            continue
        fi
        
        ((total_takes_found++))

        local found_uuid=""
        
        if [[ -n "${uuid_mapping[$take_name]:-}" ]]; then
            found_uuid="${uuid_mapping[$take_name]}"
        else
            local capture_name="$take_name"
            if [[ "$take_name" =~ ^(.+)_[0-9]+$ ]]; then
                capture_name="${BASH_REMATCH[1]}"
                if [[ -n "${uuid_mapping[$capture_name]:-}" ]]; then
                    found_uuid="${uuid_mapping[$capture_name]}"
                fi
            fi
        fi

        if [[ -n "$found_uuid" ]]; then
            takes_to_process+=("$take_dir|$take_name|$found_uuid")
            ((takes_with_uuid++))
        else
            echo "WARNING: No UUID found for take: $take_name"
        fi
    done
    
    TOTAL_TAKES=${#all_takes_list[@]}
    
    echo "📊 All takes processing result:"
    echo "   Total takes: ${#all_takes_list[@]}"
    echo "   Directories found: $total_takes_found"
    echo "   Takes with UUID: $takes_with_uuid"
    echo "   Takes without UUID: $((total_takes_found - takes_with_uuid))"
    echo ""
    
    local takes_names=()
    for take_info in "${takes_to_process[@]}"; do
        IFS='|' read -r _tmp_dir _tmp_name _tmp_uuid <<< "$take_info"
        takes_names+=("$_tmp_name")
    done
    
    echo "🔍 Finding last completed take..."
    local last_completed_take
    last_completed_take=$(find_last_completed_take "${takes_names[@]}")
    
    local remaining_takes_to_process=()
    local start_index=0
    
    if [[ -n "$last_completed_take" ]]; then
        echo "📋 Last completed take: $last_completed_take"
        for i in "${!takes_names[@]}"; do
            if [[ "${takes_names[$i]}" == "$last_completed_take" ]]; then
                start_index=$((i + 1))
                break
            fi
        done
        echo "🚀 Starting from index: $start_index"
    else
        echo "📋 No completed takes found, starting from beginning"
    fi
    
    for ((i=start_index; i<${#takes_to_process[@]}; i++)); do
        remaining_takes_to_process+=("${takes_to_process[$i]}")
    done
    
    local total_takes_to_process=${#remaining_takes_to_process[@]}
    local excluded_count=$((takes_with_uuid - total_takes_to_process))
    
    echo "📊 Completed takes status:"
    echo "   Last completed take: ${last_completed_take:-"none"}"
    echo "   Start index: $start_index"
    echo "   Skipped takes: $excluded_count"
    echo "📋 Takes to process: $total_takes_to_process"
    
    local actual_best_dirs=0
    if [[ -d "$BEST_OUT_DIR" ]]; then
        actual_best_dirs=$(find "$BEST_OUT_DIR" -maxdepth 1 -type d | wc -l)
        actual_best_dirs=$((actual_best_dirs - 1))
    fi
    
    echo "🔍 Validation info:"
    echo "   best_ego_view_rendering actual folder count: $actual_best_dirs"
    echo "   Processing start index: $start_index"
    echo "   Skipped takes count: $excluded_count"
    
    echo ""
    echo "🎯 Final processing plan:"
    echo "   Total takes found: $total_takes_found"
    echo "   Takes with UUID: $takes_with_uuid"
    echo "   Already completed takes: $excluded_count"
    echo "   Final takes to process: $total_takes_to_process"
    echo ""
    
    if [[ $total_takes_to_process -eq 0 ]]; then
        echo "❌ No takes to process. Exiting."
        return 1
    fi
    
    echo "🔍 First take to process:"
    if [[ ${#remaining_takes_to_process[@]} -gt 0 ]]; then
        IFS='|' read -r first_take_dir first_take_name first_uuid <<< "${remaining_takes_to_process[0]}"
        echo "   Take: $first_take_name"
        echo "   UUID: $first_uuid"
        echo "   Directory: $first_take_dir"
    fi
    echo ""
    
    local current_batch=0
    local processed_count=0
    
    for take_info in "${remaining_takes_to_process[@]}"; do
        IFS='|' read -r take_dir take_name uuid <<< "$take_info"
        
        while [[ ${#PIDS[@]} -ge $BATCH_SIZE ]]; do
            local new_pids=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                else
                    ((processed_count++))
                    wait "$pid" 2>/dev/null
                    local exit_code=$?
                    if [[ $exit_code -eq 0 ]]; then
                        echo "✅ Process $pid completed successfully ($processed_count/$total_takes_to_process)"
                    elif [[ $exit_code -eq 2 ]]; then
                        echo "⚠️  Process $pid skipped due to error ($processed_count/$total_takes_to_process)"
                        echo "   Check $ERROR_ROOT folder for details"
                    else
                        echo "❌ Process $pid exited abnormally (exit code: $exit_code) ($processed_count/$total_takes_to_process)"
                        echo "   Check $ERROR_ROOT folder for details"
                    fi
                fi
            done
            PIDS=("${new_pids[@]}")
            sleep 1
        done
        
        ((current_batch++))
        echo "🚀 [$current_batch] Starting process for take: $take_name"
        
        process_single_take "$take_dir" "$take_name" "$uuid" "$current_batch" &
        local new_pid=$!
        PIDS+=("$new_pid")
        
        echo "   Process ID: $new_pid"
    done
    
    echo ""
    echo "⏳ Waiting for all processes to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid" 2>/dev/null
        local exit_code=$?
        ((processed_count++))
        if [[ $exit_code -eq 0 ]]; then
            echo "✅ Process $pid completed successfully ($processed_count/$total_takes_to_process)"
        elif [[ $exit_code -eq 2 ]]; then
            echo "⚠️  Process $pid skipped due to error ($processed_count/$total_takes_to_process)"
            echo "   Check $ERROR_ROOT folder for details"
        else
            echo "❌ Process $pid exited abnormally (exit code: $exit_code) ($processed_count/$total_takes_to_process)"
            echo "   Check $ERROR_ROOT folder for details"
        fi
    done
    
    kill "$zombie_cleaner_pid" 2>/dev/null || true
    
    echo ""
    echo "🎉 All processing complete!"
}

main "$@"
