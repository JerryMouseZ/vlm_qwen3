# Qwen3-30B vLLM Deployment Troubleshooting Guide

This comprehensive guide covers common issues and their solutions when deploying Qwen3-30B with vLLM.

## 🚨 Common Issues and Solutions

### 1. Memory and GPU Issues

#### CUDA Out of Memory
**Error**: `torch.OutOfMemoryError: CUDA out of memory`

**Symptoms**:
- Server crashes during model loading
- Error mentions specific GPU memory amounts
- Process killed by system

**Solutions**:
```bash
# Option 1: Reduce GPU memory utilization
# Edit config/model_config.yaml:
model:
  gpu_memory_utilization: 0.6  # Reduce from 0.7

# Option 2: Use more GPUs for tensor parallelism
model:
  tensor_parallel_size: 8  # Use all available GPUs

# Option 3: Reduce context length
model:
  max_model_len: 8192  # Reduce from 16384

# Option 4: Enable CPU offloading
model:
  cpu_offload_gb: 8  # Offload 8GB to CPU
```

#### Insufficient GPUs
**Error**: `ValueError: tensor_parallel_size (X) is greater than the number of available GPUs (Y)`

**Solutions**:
```bash
# Check available GPUs
nvidia-smi

# Reduce tensor parallel size in config
model:
  tensor_parallel_size: 4  # Match your GPU count

# Or use specific GPUs
hardware:
  cuda_visible_devices: "0,1,2,3"
```

#### Mixed GPU Types
**Error**: Performance issues or memory errors with different GPU models

**Solutions**:
```bash
# Use only GPUs with same memory capacity
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Same GPU types only

# Check GPU memory
nvidia-smi --query-gpu=name,memory.total --format=csv
```

### 2. Model Loading Issues

#### Model Not Found
**Error**: `OSError: Qwen/Qwen3-30B-A3B does not appear to be a valid git repository`

**Solutions**:
```bash
# Check internet connection
ping huggingface.co

# Manual download
huggingface-cli download Qwen/Qwen3-30B-A3B

# Use local path if downloaded
model:
  name: "/path/to/local/model"
```

#### Authentication Issues
**Error**: `401 Unauthorized` or access denied

**Solutions**:
```bash
# Login to Hugging Face
huggingface-cli login

# Set token environment variable
export HF_TOKEN="your_token_here"

# Check model access permissions
huggingface-cli whoami
```

#### Slow Model Loading
**Symptoms**: Model takes very long to load

**Solutions**:
```bash
# Enable HF transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Use local SSD storage
# Move model to fast storage and use local path

# Increase loading workers
server:
  max_parallel_loading_workers: 4
```

### 3. Network and API Issues

#### Port Already in Use
**Error**: `OSError: [Errno 98] Address already in use`

**Solutions**:
```bash
# Find process using port
sudo netstat -tlnp | grep 8000
sudo lsof -i :8000

# Kill the process
sudo kill -9 <PID>

# Use different port
python deploy_qwen3.py --port 8001
```

#### API Not Responding
**Symptoms**: Requests timeout or hang

**Solutions**:
```bash
# Check server status
curl http://localhost:8000/health

# Check server logs
tail -f logs/vllm_server.log

# Restart with debug logging
python deploy_qwen3.py --log-level DEBUG
```

#### Slow API Responses
**Symptoms**: High latency, slow token generation

**Solutions**:
```yaml
# Optimize batch processing
model:
  max_num_batched_tokens: 8192
  max_num_seqs: 256
  enable_chunked_prefill: true

# Reduce precision if acceptable
model:
  dtype: "float16"  # or "bfloat16"
```

### 4. Performance Issues

#### Low Throughput
**Symptoms**: Fewer tokens per second than expected

**Solutions**:
```yaml
# Enable performance optimizations
model:
  enable_prefix_caching: true
  enable_chunked_prefill: true
  
# Optimize KV cache
model:
  kv_cache_dtype: "fp8"  # If supported
  block_size: 32  # Increase from 16

# Use optimal tensor parallel size
model:
  tensor_parallel_size: 4  # Usually optimal for 30B models
```

#### High Memory Usage
**Symptoms**: System runs out of RAM

**Solutions**:
```bash
# Monitor memory usage
watch -n 1 'free -h && nvidia-smi'

# Reduce batch size
model:
  max_num_seqs: 64  # Reduce from 256

# Enable swap
sudo swapon /swapfile
```

### 5. Installation Issues

#### vLLM Installation Fails
**Error**: Compilation errors during pip install

**Solutions**:
```bash
# Use pre-built wheels
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu118

# Install dependencies first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm

# Use conda environment
conda create -n vllm python=3.10
conda activate vllm
pip install vllm
```

#### CUDA Version Mismatch
**Error**: CUDA version incompatibility

**Solutions**:
```bash
# Check CUDA version
nvcc --version
nvidia-smi

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Use Docker for consistent environment
docker run --gpus all -p 8000:8000 vllm/vllm-openai
```

### 6. Configuration Issues

#### Invalid Configuration
**Error**: YAML parsing errors or invalid parameters

**Solutions**:
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/model_config.yaml'))"

# Check parameter names
python deploy_qwen3.py --help

# Use minimal config for testing
model:
  name: "Qwen/Qwen3-30B-A3B"
  tensor_parallel_size: 4
server:
  host: "0.0.0.0"
  port: 8000
```

## 🔧 Diagnostic Commands

### System Information
```bash
# GPU information
nvidia-smi
nvidia-ml-py3

# System resources
free -h
df -h
lscpu

# Python environment
python --version
pip list | grep -E "(torch|vllm|transformers)"
```

### Log Analysis
```bash
# Server logs
tail -f logs/vllm_server.log

# System logs
journalctl -u your-service-name -f

# GPU monitoring
nvidia-smi dmon -s pucvmet -d 1
```

### Performance Testing
```bash
# Basic API test
python test_api.py

# Load testing
python examples/python_client.py --mode benchmark

# Memory profiling
python -m memory_profiler deploy_qwen3.py
```

## 🚀 Performance Optimization Tips

### Hardware Optimization
1. **Use NVLink**: Ensure GPUs are connected via NVLink for better communication
2. **Fast Storage**: Use NVMe SSD for model storage
3. **Adequate RAM**: 64GB+ system memory recommended
4. **CPU**: High-core count CPU for preprocessing

### Software Optimization
1. **Batch Size**: Tune `max_num_seqs` based on your use case
2. **Context Length**: Use minimum required `max_model_len`
3. **Precision**: Use FP16 or BF16 if accuracy allows
4. **Caching**: Enable prefix caching for repeated prompts

### Monitoring
```bash
# Real-time monitoring script
#!/bin/bash
while true; do
    echo "=== $(date) ==="
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    echo "API Status: $(curl -s http://localhost:8000/health | jq -r .status)"
    sleep 5
done
```

## 📞 Getting Help

1. **Check Logs**: Always check `logs/vllm_server.log` first
2. **GitHub Issues**: Search vLLM and Qwen3 repositories
3. **Community**: Join vLLM Discord or forums
4. **Documentation**: Review official vLLM docs

## 🔗 Useful Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [Qwen3 Model Documentation](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [NVIDIA GPU Monitoring](https://developer.nvidia.com/nvidia-system-management-interface)
