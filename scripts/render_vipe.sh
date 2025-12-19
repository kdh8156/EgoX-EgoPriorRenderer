#!/bin/bash

# 사용 가능한 GPU 디바이스 설정 (,로 구분)
AVAILABLE_GPUS="0"  # 사용하고 싶은 GPU ID들을 여기에 설정

# GPU 목록을 배열로 변환
IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
echo "Using ${NUM_GPUS} GPUs: ${AVAILABLE_GPUS}"

# 실험 설정을 위한 associative array들
declare -A experiments=(
    #["cmu_bike01_2"]="cam02_static_vda_fixedcam_slammap ed3ec638-8363-4e1d-9851-c7936cbfad8c"
    
    # Cooking datasets
    # ["fair_cooking_05_2"]="cam04_static_vda_fixedcam_slammap 3cbd7070-7c55-4b15-ac31-100ab8c7298a"
    # ["georgiatech_cooking_01_01_2"]="cam01_static_vda_fixedcam_slammap 51fc36b3-e769-4617-b087-3826b280cad3"
    # #["iiith_cooking_01_1"]="cam01_static_vda_fixedcam_slammap 98f58f0f-53d6-4e41-bf41-d8d74ccbc37c" # 처음부터 사람 등장 X
    # ["indiana_cooking_01_2"]="cam03_static_vda_fixedcam_slammap 644022d7-8e50-4bb2-bab8-f7ffbfdc7d17"
    # #["minnesota_cooking_010_2"]="cam01_static_vda_fixedcam_slammap d77bb04d-c881-48be-9cc2-d781c69207cd"  # annotations 없음
    # #["nus_cooking_06_2"]="cam02_static_vda_fixedcam_slammap f7b3e85b-7681-48b3-97cb-6b0a5705022e"  # annotations 없음
    # ["sfu_cooking015_2"]="cam04_static_vda_fixedcam_slammap 9bc33576-bcb6-42a5-b040-3220456f268f"
    # ["uniandes_cooking_001_10"]="cam02_static_vda_fixedcam_slammap a46d15b2-5c90-4938-bdca-40b87f51bec1"

    #["sfu_cooking_013_1"]="cam01_static_vda_fixedcam_slammap 3059469a-03fc-4ae0-bbf7-b08187d1b290"
    #["upenn_0714_Cooking_6_2"]="gp03_static_vda_fixedcam_slammap 00bc5ad2-7c3a-403d-a4a6-e6d437d29000"
    #["sfu_cooking032_3"]="cam01_static_vda_fixedcam_slammap 5cb66fee-c010-4df7-925a-55cda04173c8"
    
    #["iiith_cooking_148_1"]="cam01_no_vda_fixedcam_slammap_exo_intr_gt 34ba42af-5c77-434e-a8c4-cf745bcaaf0d"
    # ["sfu_cooking_013_1"]="cam01_no_vda_fixedcam_slammap_exo_intr_gt 3059469a-03fc-4ae0-bbf7-b08187d1b290"
    # ["indiana_cooking_27_2"]="cam04_no_vda_fixedcam_slammap_exo_intr_gt 2f61dd27-2ad2-4029-b36d-19a02ae8feec"
    # ["uniandes_cooking_009_2"]="cam04_no_vda_fixedcam_slammap_exo_intr_gt e076b7e5-e67e-4452-8b00-83046dd85c62"
    ["georgiatech_cooking_01_03_2"]="cam02_moge_no_vda_fixedcam_slammap_exo_intr_gt 0e015bb1-8406-4f3f-a49e-3c8dd4a025e9"
)

# 공통 설정
OUTPUT_DIR="ego_view_rendering_inthewild"
POINT_SIZE="5.0"
START_FRAME="0"
END_FRAME="48"
#15373+290=15663
# SOURCE_START_FRAME="18685"
# SOURCE_END_FRAME="18733"
SOURCE_START_FRAME="0"
SOURCE_END_FRAME="299"

# 실험 실행 함수 (특정 GPU에서 실행)
run_experiment() {
    local take_name=$1
    local result_subdir=$2
    local ego_uuid=$3
    local gpu_id=$4
    
    # 경로 설정
    # INPUT_DIR="vipe_results/${take_name}/${result_subdir}"
    # EGO_CAMERA_POSE_PATH="/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/annotations/ego_pose/train/camera_pose/${ego_uuid}.json"
    # EXO_CAMERA_POSE_PATH="/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/takes/${take_name}/trajectory/gopro_calibs.csv"
    # ONLINE_CALIBRATION_PATH="/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/takes/${take_name}/trajectory/online_calibration.jsonl"

    INPUT_DIR="vipe_results/ironman_old_moge_static_vda_fixedcam_slammap"
    EGO_CAMERA_POSE_PATH="/home/nas_main/taewoongkang/dohyeon/Exo-to-Ego/Ego-Renderer-from-ViPE/inthewild_dataset/ego_prior_datasets_split_with_camera_params.json"
    EXO_CAMERA_POSE_PATH="inthewild"
    #ONLINE_CALIBRATION_PATH="/home/nas_main/kinamkim/DATA/Ego4D/dataset_train/takes/fair_cooking_05_2/trajectory/online_calibration.jsonl" #! hardcoded
    ONLINE_CALIBRATION_PATH="/home/nas5/kinamkim/DATA/tmp_EgoExo4D/takes/fair_cooking_05_2/trajectory/online_calibration.jsonl"

    echo "[GPU $gpu_id] =========================================="
    echo "[GPU $gpu_id] Running experiment: ${take_name}/${result_subdir}"
    echo "[GPU $gpu_id] =========================================="
    
    # 특정 GPU에서만 실행되도록 CUDA_VISIBLE_DEVICES 설정
    CUDA_VISIBLE_DEVICES=$gpu_id python scripts/render_vipe_pointcloud.py \
        --input_dir $INPUT_DIR \
        --out_dir $OUTPUT_DIR \
        --ego_camera_pose_path $EGO_CAMERA_POSE_PATH \
        --exo_camera_pose_path $EXO_CAMERA_POSE_PATH \
        --online_calibration_path $ONLINE_CALIBRATION_PATH \
        --point_size $POINT_SIZE \
        --start_frame $START_FRAME \
        --end_frame $END_FRAME \
        --source_start_frame $SOURCE_START_FRAME \
        --source_end_frame $SOURCE_END_FRAME \
        --fish_eye_rendering \
        --use_mean_bg \
        #--only_bg
        
    echo "[GPU $gpu_id] Completed: ${take_name}/${result_subdir}"
}

# 백그라운드 프로세스 관리를 위한 배열
declare -a BACKGROUND_PIDS=()

# 모든 실험을 GPU에 분산해서 실행
gpu_index=0
for take_name in "${!experiments[@]}"; do
    
    # 공백으로 분리해서 파라미터 추출
    IFS=' ' read -r result_subdir ego_uuid <<< "${experiments[$take_name]}"
    
    # 현재 GPU 선택 (라운드 로빈 방식)
    current_gpu=${GPU_ARRAY[$gpu_index]}
    
    echo "Starting experiment $take_name on GPU $current_gpu (index: $gpu_index)"
    
    # 백그라운드에서 실험 실행
    run_experiment "$take_name" "$result_subdir" "$ego_uuid" "$current_gpu" &
    
    # 백그라운드 프로세스 PID 저장
    BACKGROUND_PIDS+=($!)
    
    # 다음 GPU로 이동 (라운드 로빈)
    gpu_index=$(( (gpu_index + 1) % NUM_GPUS ))
    
    # GPU 메모리가 부족할 수 있으므로 약간의 지연 추가
    sleep 2
done

echo "All experiments started in parallel across $NUM_GPUS GPUs"
echo "Waiting for all processes to complete..."

# 모든 백그라운드 프로세스가 완료될 때까지 대기
for pid in "${BACKGROUND_PIDS[@]}"; do
    wait $pid
    echo "Process $pid completed"
done

echo "All experiments completed!"