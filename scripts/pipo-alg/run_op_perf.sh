#!/bin/bash
cd "$(dirname "$0")/../../" || exit 1

# Defaults
BUILD_TYPE="release"
MODE="perf"
MODEL_PATH="/home/hitori/code/impl_ai/model/Qwen3-14B-Q4_K_M.gguf"
ALG="dp"
ALG_EXTRA_ARG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        release|debug)
            BUILD_TYPE="$1"
            shift
            ;;
        alg)
            MODE="alg"            
            shift
            ;;
        perf)
            MODE="perf"
            shift
            ;;
        dp)
            ALG="dp"
            shift
            ;;
        grd)
            ALG="greedy"
            shift
            ;;
        *)
            # Assume it is the model path
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

# Set Binary Path and Output Path
BINARY="./build-${BUILD_TYPE}/bin/test-backend-ops-perf2"
OUTPUT_JSON="examples/pipo-alg/perf_result.json"
if [ $MODE == "alg" ]; then
    BINARY="./build-${BUILD_TYPE}/bin/pipo-alg"
    OUTPUT_JSON="examples/pipo-alg/alg_config.json"
fi


if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

# Prepare Command Arguments
CMD_ARGS="-m $MODEL_PATH"
if [ $MODE == "alg" ]; then
    CMD_ARGS="$MODEL_PATH -$ALG $ALG_EXTRA_ALG"
fi
# Prepare Log Directory and File
MODEL_FILENAME=$(basename "$MODEL_PATH")
MODEL_NAME="${MODEL_FILENAME%.*}"
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
# Logs structure: logs/{mode}/{build_type}/
LOG_DIR="logs/op_perf/${BUILD_TYPE}"
LOG_FILE="${LOG_DIR}/${CURRENT_DATE}_${MODE}_${MODEL_NAME}.log"

mkdir -p "$LOG_DIR"

echo "Build: $BUILD_TYPE"
echo "Executing: $BINARY $CMD_ARGS"
echo "Logging to: $LOG_FILE"
echo "Writting config to: $OUTPUT_JSON"

# Execute
$BINARY $CMD_ARGS 2> "$LOG_FILE" > "$OUTPUT_JSON"