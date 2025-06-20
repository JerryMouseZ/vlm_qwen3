# Getting Started with Qwen3-30B vLLM Deployment

This guide will help you get your Qwen3-30B model up and running quickly with vLLM.

## 🚀 Quick Start (5 Minutes)

### Prerequisites Check
```bash
# Check if you have the required hardware
nvidia-smi  # Should show 4+ GPUs with 24GB+ VRAM each

# Check Python version
python3 --version  # Should be 3.8+

# Check available disk space
df -h  # Should have 100GB+ free space
```

### One-Command Setup
```bash
# Clone and start (if you haven't already)
git clone <this-repo>
cd vlm_qwen3

# Install dependencies and start server
./quick_start.sh
```

### Verify Installation
```bash
# Test the API (in a new terminal)
curl http://localhost:8000/health

# Run comprehensive tests
python test_api.py
```

## 📖 Step-by-Step Guide

### Step 1: Environment Setup

#### Install Dependencies
```bash
# Create virtual environment (recommended)
python3 -m venv vllm_env
source vllm_env/bin/activate

# Install required packages
pip install vllm transformers accelerate pyyaml gputil aiohttp

# Verify installation
python -c "import vllm; print('vLLM version:', vllm.__version__)"
```

#### Configure Hugging Face Access
```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login (you'll need a Hugging Face account)
huggingface-cli login

# Verify access to Qwen3 model
huggingface-cli download Qwen/Qwen3-30B-A3B --dry-run
```

### Step 2: Configuration

#### Review Hardware Configuration
```bash
# Check your GPU setup
nvidia-smi

# Edit config/model_config.yaml to match your hardware:
# - Set tensor_parallel_size to your GPU count (4 or 8)
# - Adjust gpu_memory_utilization based on your GPU memory
# - Modify max_model_len if you need longer contexts
```

#### Basic Configuration Example
```yaml
# For 4x RTX 3090 setup
model:
  name: "Qwen/Qwen3-30B-A3B"
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.7
  max_model_len: 16384

server:
  host: "0.0.0.0"
  port: 8000
```

### Step 3: Deployment

#### Option A: Quick Start Script
```bash
# Easiest method - handles everything automatically
./quick_start.sh

# With custom options
./quick_start.sh --port 8001 --model-variant A3B
```

#### Option B: Manual Deployment
```bash
# Start the server manually
python deploy_qwen3.py --config config/model_config.yaml

# Or with command line options
python deploy_qwen3.py --port 8000 --tensor-parallel-size 4
```

#### Option C: Direct vLLM Command
```bash
# Use vLLM directly (for advanced users)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --trust-remote-code
```

### Step 4: Testing

#### Basic Health Check
```bash
# Check if server is running
curl http://localhost:8000/health

# Should return: {"status": "ok"}
```

#### Simple API Test
```bash
# Test text completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-30b",
    "prompt": "The future of AI is",
    "max_tokens": 50
  }'
```

#### Comprehensive Testing
```bash
# Run all tests
python test_api.py

# Run example scripts
bash examples/curl_examples.sh
python examples/python_client.py --mode demo
```

## 🔧 Common First-Time Issues

### Issue 1: CUDA Out of Memory
**Symptoms**: Server crashes with CUDA memory error

**Quick Fix**:
```yaml
# In config/model_config.yaml, reduce memory usage:
model:
  gpu_memory_utilization: 0.6  # Reduce from 0.7
  max_model_len: 8192          # Reduce from 16384
```

### Issue 2: Port Already in Use
**Symptoms**: "Address already in use" error

**Quick Fix**:
```bash
# Use a different port
./quick_start.sh --port 8001

# Or kill the existing process
sudo lsof -i :8000
sudo kill -9 <PID>
```

### Issue 3: Model Download Fails
**Symptoms**: Network errors or authentication issues

**Quick Fix**:
```bash
# Check internet connection
ping huggingface.co

# Re-authenticate
huggingface-cli login

# Try manual download
huggingface-cli download Qwen/Qwen3-30B-A3B
```

### Issue 4: Insufficient GPUs
**Symptoms**: "Not enough GPUs" error

**Quick Fix**:
```yaml
# In config/model_config.yaml, reduce GPU requirement:
model:
  tensor_parallel_size: 2  # Use fewer GPUs
```

## 📊 Performance Expectations

### What to Expect

| Hardware | Expected Performance |
|----------|---------------------|
| 4x RTX 3090 | 15-20 tokens/s, 2-3s first token |
| 8x RTX 3090 | 25-35 tokens/s, 1.5-2s first token |
| 4x A100 | 30-40 tokens/s, 1-1.5s first token |

### Performance Tuning
```bash
# Run benchmark to check your performance
python examples/python_client.py --mode benchmark

# If performance is low, see PERFORMANCE_GUIDE.md for optimization tips
```

## 🎯 Next Steps

### For Development
1. **Explore the API**: Try different prompts and parameters
2. **Integrate with your app**: Use the OpenAI-compatible API
3. **Optimize performance**: Follow the performance guide
4. **Monitor usage**: Set up logging and monitoring

### For Production
1. **Review the deployment checklist**: See `DEPLOYMENT_CHECKLIST.md`
2. **Set up monitoring**: Implement health checks and alerts
3. **Configure security**: Set up access controls if needed
4. **Plan for scaling**: Consider load balancing for high traffic

### Useful Commands
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Check server logs
tail -f logs/vllm_server.log

# Interactive chat session
python examples/python_client.py --mode chat

# Stop the server
pkill -f "python.*deploy_qwen3"
```

## 📚 Documentation Overview

- **[README.md](README.md)** - Main documentation and overview
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Detailed troubleshooting guide
- **[PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md)** - Performance optimization
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Production deployment checklist
- **[examples/](examples/)** - Code examples and API usage

## 🆘 Getting Help

### Self-Help Resources
1. Check the troubleshooting guide for common issues
2. Review the logs in `logs/vllm_server.log`
3. Verify your configuration matches your hardware
4. Test with minimal configuration first

### Community Resources
- [vLLM GitHub Issues](https://github.com/vllm-project/vllm/issues)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3 Model Documentation](https://huggingface.co/Qwen/Qwen3-30B-A3B)

### Quick Debugging
```bash
# Check system status
./quick_start.sh --check-system

# Validate configuration
python deploy_qwen3.py --validate-config

# Test with minimal config
python deploy_qwen3.py --minimal-config
```

## ✅ Success Checklist

- [ ] Server starts without errors
- [ ] Health check returns "ok"
- [ ] Simple completion test works
- [ ] Performance meets expectations
- [ ] All example scripts run successfully

**Congratulations!** Your Qwen3-30B deployment is ready to use. 

For advanced configuration and optimization, refer to the detailed documentation files in this repository.
