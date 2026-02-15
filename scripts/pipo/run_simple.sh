#!/bin/bash

# Navigate to the project root relative to this script
# Assuming script is in scripts/pipo/run_simple.sh
cd "$(dirname "$0")/../../" || exit 1

# Defaults
BUILD_TYPE="release"
MODEL_PATH="/home/hitori/code/impl_ai/model/Qwen3-14B-Q4_K_M.gguf"
CONFIG_PATH="examples/pipo-alg/alg_config.json"
MODE="pipo"
N_GL="10"
N_PREDICT="32"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        release|debug)
            BUILD_TYPE="$1"
            shift
            ;;
        base)
            MODE="base"
            shift
            ;;
        pipo)
            MODE="pipo"
            shift
            ;;
        -ngl)
            N_GL="$2"
            shift 2
            ;;
        -n)
            N_PREDICT="$2"
            shift 2
            ;;
        -c)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            # Assume it is the model path
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

# Set Binary Path
BINARY="./build-${BUILD_TYPE}/bin/llama-simple"

if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

# Prepare Command Arguments
CMD_ARGS="-m $MODEL_PATH -ngl $N_GL -n $N_PREDICT"
if [[ "$MODE" == "pipo" ]]; then
    CMD_ARGS="$CMD_ARGS -pipo $CONFIG_PATH"
fi

# Prepare Log Directory and File
MODEL_FILENAME=$(basename "$MODEL_PATH")
MODEL_NAME="${MODEL_FILENAME%.*}"
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
# Logs structure: logs/{mode}/{build_type}/
LOG_DIR="logs/${MODE}/${BUILD_TYPE}"
LOG_FILE="${LOG_DIR}/${CURRENT_DATE}_${MODE}_${MODEL_NAME}.log"

mkdir -p "$LOG_DIR"

echo "Mode: $MODE"
echo "Build: $BUILD_TYPE"
echo "Executing: $BINARY $CMD_ARGS"
echo "Logging to: $LOG_FILE"

# Execute
# SC2086: Double quote to prevent globbing and word splitting.
# We explicitly want splitting for CMD_ARGS here to pass as separate arguments
$BINARY $CMD_ARGS > "$LOG_FILE" 2>&1
