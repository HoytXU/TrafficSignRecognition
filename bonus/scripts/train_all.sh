#!/bin/bash
# Batch Training Script
# Trains all models with specified hyperparameters

# Define hyperparameters
EPOCH=10
LR=0.001
WEIGHT_DECAY=0.001
BATCH_SIZE=128

# Define model architectures
MODELS=("lenet" "resnet18" "vgg16" "alexnet" "squeezenet1_0" "vit_b_16" "my_net")

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BONUS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BONUS_DIR")"

# Create logs directory
mkdir -p "$BONUS_DIR/logs"

# Train each model
for MODEL in "${MODELS[@]}"
do
    LOG_FILE="$BONUS_DIR/logs/${MODEL}-${EPOCH}-${LR}-${BATCH_SIZE}-${WEIGHT_DECAY}.log"
    echo "=========================================="
    echo "Training ${MODEL}..."
    echo "Log file: ${LOG_FILE}"
    echo "=========================================="
    
    cd "$PROJECT_ROOT"
    python "$BONUS_DIR/training/train.py" \
        --epoch ${EPOCH} \
        --lr ${LR} \
        --batch_size ${BATCH_SIZE} \
        --weight_decay ${WEIGHT_DECAY} \
        --model ${MODEL} \
        > ${LOG_FILE} 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ ${MODEL} training completed successfully"
    else
        echo "✗ ${MODEL} training failed. Check log: ${LOG_FILE}"
    fi
    echo ""
done

echo "All training jobs completed!"

