#!/bin/bash
# Quick start script for Qwen3-30B vLLM deployment

set -e  # Exit on any error

echo "=== Qwen3-30B vLLM Quick Start ==="
echo "This script will deploy Qwen3-30B using vLLM"
echo ""

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "Warning: Running as root. Consider using a non-root user for security."
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python 3.8+ required, found $python_version"
    exit 1
fi

# Check if vLLM is installed
if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM not found. Please install it first:"
    echo "pip install vllm transformers accelerate"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU support may not be available."
else
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $gpu_count GPU(s)"
    if [ $gpu_count -lt 4 ]; then
        echo "Warning: Qwen3-30B works best with 4+ GPUs. You have $gpu_count."
        echo "Consider adjusting tensor_parallel_size in config/model_config.yaml"
    fi
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p config
mkdir -p examples

# Check if config exists
if [ ! -f "config/model_config.yaml" ]; then
    echo "Error: Configuration file not found at config/model_config.yaml"
    echo "Please ensure the configuration file exists."
    exit 1
fi

# Make scripts executable
chmod +x deploy_qwen3.py
chmod +x test_api.py
chmod +x examples/curl_examples.sh
chmod +x examples/python_client.py

echo ""
echo "=== Starting Deployment ==="
echo ""

# Parse command line arguments
MODEL_VARIANT="A3B"
CONFIG_FILE="config/model_config.yaml"
BACKGROUND=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-variant)
            MODEL_VARIANT="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model-variant A2B|A3B    Model variant to deploy (default: A3B)"
            echo "  --config FILE               Configuration file path (default: config/model_config.yaml)"
            echo "  --background                Run in background"
            echo "  --help                      Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Deploy A3B variant in foreground"
            echo "  $0 --model-variant A2B     # Deploy A2B variant"
            echo "  $0 --background             # Deploy in background"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Model variant: $MODEL_VARIANT"
echo "Config file: $CONFIG_FILE"
echo "Background mode: $BACKGROUND"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ ! -z "$DEPLOY_PID" ]; then
        echo "Stopping deployment process..."
        kill $DEPLOY_PID 2>/dev/null || true
        wait $DEPLOY_PID 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Start deployment
if [ "$BACKGROUND" = true ]; then
    echo "Starting deployment in background..."
    python3 deploy_qwen3.py --config "$CONFIG_FILE" --model-variant "$MODEL_VARIANT" > logs/deployment.log 2>&1 &
    DEPLOY_PID=$!
    echo "Deployment started with PID: $DEPLOY_PID"
    echo "Logs are being written to logs/deployment.log"
    echo ""
    
    # Wait for server to start
    echo "Waiting for server to start..."
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo "Server is ready!"
            break
        fi
        if ! kill -0 $DEPLOY_PID 2>/dev/null; then
            echo "Deployment process died. Check logs/deployment.log for details."
            exit 1
        fi
        sleep 5
        echo -n "."
    done
    echo ""
    
    if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server failed to start within 5 minutes. Check logs for details."
        exit 1
    fi
    
    echo "=== Deployment Complete ==="
    echo "Server URL: http://localhost:8000"
    echo "API Documentation: http://localhost:8000/docs"
    echo "Metrics: http://localhost:8000:8001/metrics"
    echo ""
    echo "To test the deployment:"
    echo "  python3 test_api.py"
    echo "  ./examples/curl_examples.sh"
    echo "  python3 examples/python_client.py --mode demo"
    echo ""
    echo "To stop the server:"
    echo "  kill $DEPLOY_PID"
    echo ""
    echo "Logs:"
    echo "  tail -f logs/deployment.log"
    echo "  tail -f logs/vllm_server.log"
    
else
    echo "Starting deployment in foreground..."
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    python3 deploy_qwen3.py --config "$CONFIG_FILE" --model-variant "$MODEL_VARIANT"
fi
