# Qwen3-30B vLLM Deployment Guide

This repository provides a comprehensive deployment solution for the Qwen3-30B model using vLLM (Virtual Large Language Model) inference engine.

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone <this-repo>
cd vlm_qwen3

# 2. Install dependencies
pip install vllm transformers accelerate pyyaml gputil aiohttp

# 3. Quick deployment
./quick_start.sh

# 4. Test the deployment
python3 test_api.py
```

> 📖 **New to this deployment?** Check out our **[Getting Started Guide](GETTING_STARTED.md)** for a detailed walkthrough!

## 📋 System Requirements

### Hardware Requirements
- **GPUs**: 4-8 NVIDIA GPUs with 24GB+ VRAM each (RTX 3090, RTX 4090, A100, etc.)
- **RAM**: 64GB+ system memory recommended
- **Storage**: 100GB+ free disk space
- **Network**: High-speed internet for model download

### Software Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8+
- **CUDA**: 11.8+ or 12.0+
- **PyTorch**: 2.0+

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │───▶│   vLLM Server   │───▶│  Qwen3-30B-A3B  │
│                 │    │   (Port 8000)   │    │   (4-8 GPUs)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
vlm_qwen3/
├── config/
│   └── model_config.yaml      # Model configuration
├── examples/
│   ├── curl_examples.sh       # cURL API examples
│   └── python_client.py       # Python client examples
├── logs/                      # Log files
├── deploy_qwen3.py           # Main deployment script
├── test_api.py               # API testing script
├── quick_start.sh            # Quick start script
├── README.md                 # This file
├── GETTING_STARTED.md        # Step-by-step guide for beginners
├── TROUBLESHOOTING.md        # Comprehensive troubleshooting guide
├── PERFORMANCE_GUIDE.md      # Performance optimization guide
└── DEPLOYMENT_CHECKLIST.md   # Production deployment checklist
```

## ⚙️ Configuration

### Model Configuration (`config/model_config.yaml`)

Key parameters to adjust based on your hardware:

```yaml
model:
  name: "Qwen/Qwen3-30B-A3B"
  tensor_parallel_size: 4      # Number of GPUs to use
  gpu_memory_utilization: 0.85 # GPU memory usage (0.7-0.9)
  max_model_len: 32768         # Context length
  
server:
  host: "0.0.0.0"
  port: 8000                   # Change if port 8000 is in use
```

## 🚀 Deployment Options

### Option 1: Quick Start (Recommended)
```bash
./quick_start.sh --model-variant A3B
```

### Option 2: Manual Deployment
```bash
python3 deploy_qwen3.py --config config/model_config.yaml --model-variant A3B
```

### Option 3: Direct vLLM Command
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --trust-remote-code
```

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/health
```

### Simple Completion
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### Comprehensive Testing
```bash
# Run all tests
python3 test_api.py

# Run specific tests
python3 test_api.py --test health
python3 test_api.py --test completions
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
**Error**: `torch.OutOfMemoryError: CUDA out of memory`

**Solutions**:
- Increase `tensor_parallel_size` to use more GPUs
- Reduce `gpu_memory_utilization` (try 0.7 or 0.6)
- Reduce `max_model_len` (try 16384 or 8192)
- Use quantized models (GPTQ/AWQ versions)

#### 2. Port Already in Use
**Error**: `OSError: [Errno 98] Address already in use`

**Solutions**:
```bash
# Check what's using the port
netstat -tlnp | grep 8000

# Kill the process or use a different port
./quick_start.sh --port 8001
```

#### 3. Model Not Found
**Error**: Model download fails or not found

**Solutions**:
- Check internet connection
- Verify Hugging Face access
- Try manual download:
```bash
huggingface-cli download Qwen/Qwen3-30B-A3B
```

#### 4. Insufficient GPUs
**Error**: Not enough GPUs available

**Solutions**:
- Reduce `tensor_parallel_size` in config
- Use fewer GPUs (minimum 2 for 30B model)
- Consider using a smaller model variant

### Memory Optimization

For systems with limited GPU memory:

```yaml
# In config/model_config.yaml
model:
  tensor_parallel_size: 8      # Use all available GPUs
  gpu_memory_utilization: 0.7  # Reduce memory usage
  max_model_len: 16384         # Reduce context length
  swap_space: 8                # Increase swap space
```

### Performance Tuning

For better performance:

```yaml
model:
  enable_chunked_prefill: true
  max_num_batched_tokens: 4096
  max_num_seqs: 128
```

## 📊 Performance Benchmarks

Expected performance on different hardware configurations:

| GPUs | Memory | Throughput | Latency |
|------|--------|------------|---------|
| 4x RTX 3090 | 96GB | ~15 tokens/s | ~2s |
| 8x RTX 3090 | 192GB | ~25 tokens/s | ~1.5s |
| 4x A100 | 320GB | ~30 tokens/s | ~1s |

## 🔌 API Usage

### OpenAI-Compatible API

The deployed model provides an OpenAI-compatible API:

```python
import openai

client = openai.OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="qwen3-30b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

### Available Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List available models
- `POST /v1/completions` - Text completion
- `POST /v1/chat/completions` - Chat completion
- `GET /docs` - API documentation

## 🛠️ Advanced Configuration

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### Custom Model Paths

To use a local model:

```yaml
model:
  name: "/path/to/local/model"
  trust_remote_code: true
```

### Quantization Support

For quantized models:

```yaml
model:
  name: "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
  quantization: "gptq"
```

## 📝 Examples

See the `examples/` directory for:
- `curl_examples.sh` - cURL command examples
- `python_client.py` - Python client examples

## 📚 Additional Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** - Step-by-step guide for first-time users
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Comprehensive troubleshooting guide for common issues
- **[PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md)** - Detailed performance optimization strategies
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Production deployment checklist

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review vLLM documentation
3. Check Qwen3 model documentation
4. Open an issue in this repository

## 🔗 Useful Links

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
