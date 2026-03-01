#!/usr/bin/env bash
# Shell script to train VQ-VAE model in two stages
# Stage 1: Train VQ-VAE (encoder + vector quantizer + decoder)
# Stage 2: Train PixelCNN Prior on discrete codes

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Data path (REQUIRED - update this to your MIMIC-IV-ECG path)
DATA_DIR="${DATA_DIR:-/work/vajira/DATA/SEARCH/MIMIC_IV_ECG_raw_v1/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0}"

# Experiment settings
EXP_NAME_STAGE1="${EXP_NAME_STAGE1:-vqvae_mimic_standalone}"
EXP_NAME_STAGE2="${EXP_NAME_STAGE2:-prior_mimic_standalone}"
SEED="${SEED:-42}"
RUNS_ROOT="${RUNS_ROOT:-runs}"

# Data settings
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"  # Set to null for full dataset
VAL_SPLIT="${VAL_SPLIT:-0.1}"
TEST_SPLIT="${TEST_SPLIT:-0.1}"

# Model settings (Stage 1)
IN_CHANNELS="${IN_CHANNELS:-12}"
BASE_CHANNELS="${BASE_CHANNELS:-64}"
LATENT_CHANNELS="${LATENT_CHANNELS:-64}"
NUM_RES_BLOCKS="${NUM_RES_BLOCKS:-2}"
NUM_EMBEDDINGS="${NUM_EMBEDDINGS:-512}"
COMMITMENT_COST="${COMMITMENT_COST:-0.25}"
SEQ_LENGTH="${SEQ_LENGTH:-5000}"

# Model settings (Stage 2)
HIDDEN_DIM="${HIDDEN_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-3}"

# Training settings
LR_STAGE1="${LR_STAGE1:-0.0001}"
LR_STAGE2="${LR_STAGE2:-0.001}"
MAX_EPOCHS_STAGE1="${MAX_EPOCHS_STAGE1:-100}"
MAX_EPOCHS_STAGE2="${MAX_EPOCHS_STAGE2:-100}"
ACCELERATOR="${ACCELERATOR:-gpu}"
DEVICES="${DEVICES:-0}"
LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-50}"
CHECK_VAL_EVERY_N_EPOCH="${CHECK_VAL_EVERY_N_EPOCH:-1}"
GRADIENT_CLIP="${GRADIENT_CLIP:-1.0}"
PATIENCE="${PATIENCE:-10}"
SAVE_TOP_K="${SAVE_TOP_K:-3}"

# Visualization settings (Stage 1)
VIZ_EVERY_N_EPOCHS="${VIZ_EVERY_N_EPOCHS:-5}"
VIZ_NUM_SAMPLES="${VIZ_NUM_SAMPLES:-4}"

# Weights & Biases settings
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-ecg-vqvae}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_TAGS="${WANDB_TAGS:-}"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

# ============================================================================
# Validation
# ============================================================================

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    print_error "Data directory not found: ${DATA_DIR}"
    print_error "Please set DATA_DIR environment variable or update the script"
    exit 1
fi

# Check if training script exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_vqvae_standalone.py"

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    print_error "Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi

# ============================================================================
# Stage Selection
# ============================================================================

STAGE="${1:-both}"

if [[ ! "${STAGE}" =~ ^(1|2|both)$ ]]; then
    print_error "Invalid stage: ${STAGE}"
    echo "Usage: $0 [1|2|both]"
    echo "  1    - Train VQ-VAE only (Stage 1)"
    echo "  2    - Train Prior only (Stage 2) - requires VQ-VAE checkpoint"
    echo "  both - Train both stages sequentially (default)"
    exit 1
fi

# ============================================================================
# Stage 1: Train VQ-VAE
# ============================================================================

if [[ "${STAGE}" == "1" || "${STAGE}" == "both" ]]; then
    print_header "STAGE 1: Training VQ-VAE"
    
    print_info "Experiment: ${EXP_NAME_STAGE1}"
    print_info "Data directory: ${DATA_DIR}"
    print_info "Batch size: ${BATCH_SIZE}"
    print_info "Max samples: ${MAX_SAMPLES}"
    print_info "Max epochs: ${MAX_EPOCHS_STAGE1}"
    print_info "Learning rate: ${LR_STAGE1}"
    print_info "Codebook size: ${NUM_EMBEDDINGS}"
    print_info "Wandb enabled: ${WANDB_ENABLED}"
    
    # Build command
    CMD="python ${TRAIN_SCRIPT} \
        --stage 1 \
        --exp-name ${EXP_NAME_STAGE1} \
        --seed ${SEED} \
        --runs-root ${RUNS_ROOT} \
        --data-dir ${DATA_DIR} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --val-split ${VAL_SPLIT} \
        --test-split ${TEST_SPLIT} \
        --in-channels ${IN_CHANNELS} \
        --base-channels ${BASE_CHANNELS} \
        --latent-channels ${LATENT_CHANNELS} \
        --num-res-blocks ${NUM_RES_BLOCKS} \
        --num-embeddings ${NUM_EMBEDDINGS} \
        --commitment-cost ${COMMITMENT_COST} \
        --seq-length ${SEQ_LENGTH} \
        --lr ${LR_STAGE1} \
        --max-epochs ${MAX_EPOCHS_STAGE1} \
        --accelerator ${ACCELERATOR} \
        --devices ${DEVICES} \
        --log-every-n-steps ${LOG_EVERY_N_STEPS} \
        --check-val-every-n-epoch ${CHECK_VAL_EVERY_N_EPOCH} \
        --gradient-clip ${GRADIENT_CLIP} \
        --patience ${PATIENCE} \
        --save-top-k ${SAVE_TOP_K} \
        --viz-every-n-epochs ${VIZ_EVERY_N_EPOCHS} \
        --viz-num-samples ${VIZ_NUM_SAMPLES}"
    
    # Add wandb if enabled
    if [ "${WANDB_ENABLED}" = "true" ]; then
        CMD="${CMD} --wandb"
        CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
        
        if [ -n "${WANDB_ENTITY}" ]; then
            CMD="${CMD} --wandb-entity ${WANDB_ENTITY}"
        fi
        
        if [ -n "${WANDB_RUN_NAME}" ]; then
            CMD="${CMD} --wandb-run-name ${WANDB_RUN_NAME}"
        fi
        
        if [ -n "${WANDB_TAGS}" ]; then
            CMD="${CMD} --wandb-tags ${WANDB_TAGS}"
        fi
    fi
    
    # Add max-samples if specified
    if [ -n "${MAX_SAMPLES}" ] && [ "${MAX_SAMPLES}" != "null" ]; then
        CMD="${CMD} --max-samples ${MAX_SAMPLES}"
    fi
    
    print_info "Running command:"
    echo "${CMD}"
    echo ""
    
    # Run training
    eval "${CMD}"
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        print_error "Stage 1 training failed"
        exit 1
    fi
    
    # Find best checkpoint
    CHECKPOINT_DIR="${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/checkpoints"
    BEST_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -name "epoch*.ckpt" -type f | sort | tail -n 1)
    
    if [ -z "${BEST_CHECKPOINT}" ]; then
        BEST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"
    fi
    
    print_header "Stage 1 Complete"
    print_info "Best checkpoint: ${BEST_CHECKPOINT}"
    
    # Export for Stage 2
    export VQVAE_CHECKPOINT="${BEST_CHECKPOINT}"
fi

# ============================================================================
# Stage 2: Train Prior
# ============================================================================

if [[ "${STAGE}" == "2" || "${STAGE}" == "both" ]]; then
    print_header "STAGE 2: Training PixelCNN Prior"
    
    # Check if VQ-VAE checkpoint is provided
    if [ -z "${VQVAE_CHECKPOINT:-}" ]; then
        print_error "VQ-VAE checkpoint not specified"
        print_error "Please set VQVAE_CHECKPOINT environment variable or run Stage 1 first"
        exit 1
    fi
    
    if [ ! -f "${VQVAE_CHECKPOINT}" ]; then
        print_error "VQ-VAE checkpoint not found: ${VQVAE_CHECKPOINT}"
        exit 1
    fi
    
    print_info "Experiment: ${EXP_NAME_STAGE2}"
    print_info "VQ-VAE checkpoint: ${VQVAE_CHECKPOINT}"
    print_info "Data directory: ${DATA_DIR}"
    print_info "Batch size: ${BATCH_SIZE}"
    print_info "Max samples: ${MAX_SAMPLES}"
    print_info "Max epochs: ${MAX_EPOCHS_STAGE2}"
    print_info "Learning rate: ${LR_STAGE2}"
    print_info "Wandb enabled: ${WANDB_ENABLED}"
    
    # Build command
    CMD="python ${TRAIN_SCRIPT} \
        --stage 2 \
        --exp-name ${EXP_NAME_STAGE2} \
        --seed ${SEED} \
        --runs-root ${RUNS_ROOT} \
        --data-dir ${DATA_DIR} \
        --batch-size ${BATCH_SIZE} \
        --num-workers ${NUM_WORKERS} \
        --val-split ${VAL_SPLIT} \
        --test-split ${TEST_SPLIT} \
        --num-embeddings ${NUM_EMBEDDINGS} \
        --hidden-dim ${HIDDEN_DIM} \
        --num-layers ${NUM_LAYERS} \
        --seq-length ${SEQ_LENGTH} \
        --lr ${LR_STAGE2} \
        --max-epochs ${MAX_EPOCHS_STAGE2} \
        --accelerator ${ACCELERATOR} \
        --devices ${DEVICES} \
        --log-every-n-steps ${LOG_EVERY_N_STEPS} \
        --check-val-every-n-epoch ${CHECK_VAL_EVERY_N_EPOCH} \
        --gradient-clip ${GRADIENT_CLIP} \
        --patience ${PATIENCE} \
        --save-top-k ${SAVE_TOP_K} \
        --vqvae-checkpoint ${VQVAE_CHECKPOINT}"
    
    # Add wandb if enabled
    if [ "${WANDB_ENABLED}" = "true" ]; then
        CMD="${CMD} --wandb"
        CMD="${CMD} --wandb-project ${WANDB_PROJECT}"
        
        if [ -n "${WANDB_ENTITY}" ]; then
            CMD="${CMD} --wandb-entity ${WANDB_ENTITY}"
        fi
        
        if [ -n "${WANDB_RUN_NAME}" ]; then
            CMD="${CMD} --wandb-run-name ${WANDB_RUN_NAME}_stage2"
        fi
        
        if [ -n "${WANDB_TAGS}" ]; then
            CMD="${CMD} --wandb-tags ${WANDB_TAGS}"
        fi
    fi
    
    # Add max-samples if specified
    if [ -n "${MAX_SAMPLES}" ] && [ "${MAX_SAMPLES}" != "null" ]; then
        CMD="${CMD} --max-samples ${MAX_SAMPLES}"
    fi
    
    print_info "Running command:"
    echo "${CMD}"
    echo ""
    
    # Run training
    eval "${CMD}"
    
    # Check if training succeeded
    if [ $? -ne 0 ]; then
        print_error "Stage 2 training failed"
        exit 1
    fi
    
    # Find best checkpoint
    CHECKPOINT_DIR="${RUNS_ROOT}/${EXP_NAME_STAGE2}/seed_${SEED}/checkpoints"
    BEST_CHECKPOINT=$(find "${CHECKPOINT_DIR}" -name "epoch*.ckpt" -type f | sort | tail -n 1)
    
    if [ -z "${BEST_CHECKPOINT}" ]; then
        BEST_CHECKPOINT="${CHECKPOINT_DIR}/last.ckpt"
    fi
    
    print_header "Stage 2 Complete"
    print_info "Best checkpoint: ${BEST_CHECKPOINT}"
fi

# ============================================================================
# Summary
# ============================================================================

print_header "Training Complete!"

if [[ "${STAGE}" == "1" || "${STAGE}" == "both" ]]; then
    echo "Stage 1 (VQ-VAE) results:"
    echo "  - Checkpoints: ${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/checkpoints/"
    echo "  - Samples: ${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/samples/"
    echo "  - TensorBoard logs: ${RUNS_ROOT}/${EXP_NAME_STAGE1}/seed_${SEED}/tb/"
    echo ""
fi

if [[ "${STAGE}" == "2" || "${STAGE}" == "both" ]]; then
    echo "Stage 2 (Prior) results:"
    echo "  - Checkpoints: ${RUNS_ROOT}/${EXP_NAME_STAGE2}/seed_${SEED}/checkpoints/"
    echo "  - TensorBoard logs: ${RUNS_ROOT}/${EXP_NAME_STAGE2}/seed_${SEED}/tb/"
    echo ""
fi

echo "To view training progress:"
echo "  TensorBoard: tensorboard --logdir=${RUNS_ROOT}"
if [ "${WANDB_ENABLED}" = "true" ]; then
    echo "  Weights & Biases: https://wandb.ai/${WANDB_ENTITY:-your-username}/${WANDB_PROJECT}"
fi
echo ""

if [[ "${STAGE}" == "both" ]]; then
    echo "Both stages completed successfully!"
    echo "You can now use the trained models for ECG generation."
fi
