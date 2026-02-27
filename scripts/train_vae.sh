#!/bin/bash

# VAE Training Script for ECGEN
# Usage: bash scripts/train_vae.sh [options]

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Data paths
DATA_DIR="/path/to/mimic-iv-ecg"  # CHANGE THIS to your MIMIC-IV-ECG path
OUTPUT_DIR="runs/vae"
EXP_NAME="vae_mimic_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
BATCH_SIZE=32
NUM_WORKERS=4
MAX_EPOCHS=100
LEARNING_RATE=0.0001
KL_WEIGHT=0.0001

# Model architecture
IN_CHANNELS=12
BASE_CHANNELS=64
LATENT_CHANNELS=8
NUM_RES_BLOCKS=2

# Data splits
VAL_SPLIT=0.1
TEST_SPLIT=0.1

# Training settings
GPUS=1
SEED=42
GRADIENT_CLIP=1.0
PATIENCE=10
SAVE_TOP_K=3

# Optional: limit samples for debugging
MAX_SAMPLES=""  # Set to a number like 1000 for quick testing

# Resume from checkpoint (optional)
RESUME=""  # Set to checkpoint path to resume training

# ============================================================================
# Parse command line arguments (optional)
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --exp_name)
            EXP_NAME="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash scripts/train_vae.sh [options]"
            echo ""
            echo "Options:"
            echo "  --data_dir PATH          Path to MIMIC-IV-ECG data directory"
            echo "  --output_dir PATH        Output directory (default: runs/vae)"
            echo "  --exp_name NAME          Experiment name"
            echo "  --batch_size N           Batch size (default: 32)"
            echo "  --max_epochs N           Maximum epochs (default: 100)"
            echo "  --learning_rate LR       Learning rate (default: 0.0001)"
            echo "  --gpus N                 Number of GPUs (default: 1)"
            echo "  --seed N                 Random seed (default: 42)"
            echo "  --resume PATH            Resume from checkpoint"
            echo "  --max_samples N          Limit number of samples (for debugging)"
            echo "  --help                   Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Validation
# ============================================================================

if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory not found: $DATA_DIR"
    echo "Please set DATA_DIR to your MIMIC-IV-ECG path"
    exit 1
fi

if [ ! -f "$DATA_DIR/machine_measurements.csv" ]; then
    echo "Error: machine_measurements.csv not found in $DATA_DIR"
    echo "Please ensure you have the complete MIMIC-IV-ECG dataset"
    exit 1
fi

# ============================================================================
# Setup
# ============================================================================

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Print Configuration
# ============================================================================

echo "============================================================================"
echo "VAE Training Configuration"
echo "============================================================================"
echo "Project root:      $PROJECT_ROOT"
echo "Data directory:    $DATA_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo "Experiment name:   $EXP_NAME"
echo ""
echo "Training Settings:"
echo "  Batch size:      $BATCH_SIZE"
echo "  Max epochs:      $MAX_EPOCHS"
echo "  Learning rate:   $LEARNING_RATE"
echo "  KL weight:       $KL_WEIGHT"
echo "  GPUs:            $GPUS"
echo "  Seed:            $SEED"
echo ""
echo "Model Architecture:"
echo "  Input channels:  $IN_CHANNELS"
echo "  Base channels:   $BASE_CHANNELS"
echo "  Latent channels: $LATENT_CHANNELS"
echo "  Res blocks:      $NUM_RES_BLOCKS"
echo ""
if [ -n "$RESUME" ]; then
    echo "Resuming from:     $RESUME"
fi
if [ -n "$MAX_SAMPLES" ]; then
    echo "Max samples:       $MAX_SAMPLES (DEBUG MODE)"
fi
echo "============================================================================"
echo ""

# ============================================================================
# Build command
# ============================================================================

CMD="python $SCRIPT_DIR/train_vae_full.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --exp_name $EXP_NAME \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --max_epochs $MAX_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --kl_weight $KL_WEIGHT \
    --in_channels $IN_CHANNELS \
    --base_channels $BASE_CHANNELS \
    --latent_channels $LATENT_CHANNELS \
    --num_res_blocks $NUM_RES_BLOCKS \
    --val_split $VAL_SPLIT \
    --test_split $TEST_SPLIT \
    --gpus $GPUS \
    --seed $SEED \
    --gradient_clip $GRADIENT_CLIP \
    --patience $PATIENCE \
    --save_top_k $SAVE_TOP_K"

if [ -n "$RESUME" ]; then
    CMD="$CMD --resume $RESUME"
fi

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# ============================================================================
# Run training
# ============================================================================

echo "Starting training..."
echo "Command: $CMD"
echo ""

$CMD

# ============================================================================
# Completion
# ============================================================================

echo ""
echo "============================================================================"
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR/$EXP_NAME"
echo "============================================================================"
