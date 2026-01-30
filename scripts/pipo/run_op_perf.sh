#!/bin/bash
cd "$(dirname "$0")/../../" || exit 1

# Defaults
BUILD_TYPE="release"
MODEL_PATH="/home/hitori/code/impl_ai/model/Qwen3-14B-Q4_K_M.gguf"
CONFIG_PATH="scripts/pipo/alg_config.json"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        release|debug)
            BUILD_TYPE="$1"
            shift
            ;;
        *)
            # Assume it is the model path
            MODEL_PATH="$1"
            shift
            ;;
    esac
done

# Set Binary Path
BINARY="./build-${BUILD_TYPE}/bin/test-backend-ops-perf2"

if [[ ! -f "$BINARY" ]]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

# Prepare Command Arguments
CMD_ARGS="-m $MODEL_PATH"

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
echo "Writting config to: $CONFIG_PATH"

# Execute
# SC2086: Double quote to prevent globbing and word splitting.
# We explicitly want splitting for CMD_ARGS here to pass as separate arguments
$BINARY $CMD_ARGS 2> "$LOG_FILE" > $CONFIG_PATH



# ./build-release/bin/test-backend-ops-perf2 -m ../model/Qwen3-14B-Q4_K_M.gguf > scripts/pipo/alg_config.json 2> logs/pipo/release/op_perf_log.log 