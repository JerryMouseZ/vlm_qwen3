# Qwen3-30B Performance Optimization Guide

This guide provides detailed strategies for optimizing the performance of your Qwen3-30B deployment.

## 🎯 Performance Targets

### Expected Performance Baselines

| Configuration | Throughput | Latency | Memory Usage |
|---------------|------------|---------|--------------|
| 4x RTX 3090 (24GB) | 15-20 tokens/s | 2-3s first token | ~90GB GPU |
| 8x RTX 3090 (24GB) | 25-35 tokens/s | 1.5-2s first token | ~180GB GPU |
| 4x A100 (80GB) | 30-40 tokens/s | 1-1.5s first token | ~120GB GPU |
| 8x A100 (80GB) | 50-70 tokens/s | 0.8-1.2s first token | ~240GB GPU |

## ⚙️ Configuration Optimization

### 1. Memory Configuration

#### Optimal GPU Memory Utilization
```yaml
model:
  # Conservative setting for stability
  gpu_memory_utilization: 0.7
  
  # Aggressive setting for maximum performance
  gpu_memory_utilization: 0.85
  
  # Safe setting for mixed workloads
  gpu_memory_utilization: 0.6
```

#### KV Cache Optimization
```yaml
model:
  # Use smaller data type for KV cache
  kv_cache_dtype: "fp8"  # Saves ~50% KV cache memory
  
  # Optimize block size
  block_size: 32  # Larger blocks = better memory efficiency
  
  # Enable swap for overflow
  swap_space: 8  # 8GB swap space
```

### 2. Parallelism Configuration

#### Tensor Parallelism
```yaml
model:
  # For 4 GPUs (recommended for 30B models)
  tensor_parallel_size: 4
  
  # For 8 GPUs (better memory distribution)
  tensor_parallel_size: 8
  
  # Rule of thumb: Use all available GPUs for 30B+ models
```

#### Pipeline Parallelism
```yaml
model:
  # Keep at 1 for simplicity (recommended)
  pipeline_parallel_size: 1
  
  # Use only if you have many GPUs (16+)
  pipeline_parallel_size: 2
```

### 3. Batch Processing Optimization

#### Batch Size Tuning
```yaml
model:
  # Conservative (low latency)
  max_num_seqs: 64
  max_num_batched_tokens: 4096
  
  # Balanced (good throughput/latency)
  max_num_seqs: 128
  max_num_batched_tokens: 8192
  
  # Aggressive (high throughput)
  max_num_seqs: 256
  max_num_batched_tokens: 16384
```

#### Chunked Prefill
```yaml
model:
  # Enable for better batching
  enable_chunked_prefill: true
  
  # Tune chunk size
  max_num_batched_tokens: 8192  # Adjust based on GPU memory
```

### 4. Advanced Optimizations

#### Prefix Caching
```yaml
model:
  # Enable for repeated prompt patterns
  enable_prefix_caching: true
  
  # Useful for chat applications with system prompts
```

#### Attention Optimizations
```yaml
model:
  # Use optimized attention backend (if available)
  attention_backend: "FLASHINFER"  # Check vLLM version support
  
  # Alternative backends
  attention_backend: "XFORMERS"
  attention_backend: "FLASH_ATTN"
```

## 🔧 Hardware Optimization

### 1. GPU Configuration

#### Optimal GPU Setup
```bash
# Use GPUs with NVLink for better communication
nvidia-smi topo -m

# Set optimal GPU clocks
sudo nvidia-smi -pm 1  # Persistence mode
sudo nvidia-smi -ac 1215,1410  # Memory,Graphics clocks (adjust for your GPU)

# Optimize GPU power limits
sudo nvidia-smi -pl 350  # Set power limit (watts)
```

#### Memory Bandwidth Optimization
```bash
# Enable ECC if available (slight performance cost but better reliability)
sudo nvidia-smi -e 1

# Optimize memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,roundup_power2_divisions:16
```

### 2. System Configuration

#### CPU Optimization
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Set CPU affinity for NUMA optimization
numactl --cpubind=0 --membind=0 python deploy_qwen3.py
```

#### Memory Configuration
```bash
# Increase shared memory
echo 'kernel.shmmax = 68719476736' | sudo tee -a /etc/sysctl.conf
echo 'kernel.shmall = 4294967296' | sudo tee -a /etc/sysctl.conf

# Optimize memory allocation
echo 'vm.overcommit_memory = 1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### 3. Storage Optimization

#### Model Storage
```bash
# Use NVMe SSD for model files
# Mount with optimal options
sudo mount -o noatime,nodiratime /dev/nvme0n1 /models

# Pre-load model to memory (if enough RAM)
vmtouch -t /path/to/model/
```

## 📊 Monitoring and Profiling

### 1. Real-time Monitoring

#### GPU Monitoring Script
```bash
#!/bin/bash
# save as monitor_gpu.sh
while true; do
    clear
    echo "=== GPU Status $(date) ==="
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F',' '{printf "GPU%s: %s | GPU:%s%% MEM:%s%% | %sMB/%sMB | %s°C %sW\n", $1, $2, $3, $4, $5, $6, $7, $8}'
    echo ""
    echo "=== API Status ==="
    curl -s http://localhost:8000/health | jq -r '.status // "ERROR"'
    sleep 2
done
```

#### Performance Metrics
```python
# save as performance_monitor.py
import time
import requests
import psutil
import GPUtil

def monitor_performance():
    while True:
        # API health check
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            api_status = response.json().get("status", "ERROR")
        except:
            api_status = "ERROR"
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        
        print(f"Time: {time.strftime('%H:%M:%S')}")
        print(f"API: {api_status}")
        print(f"CPU: {cpu_percent}% | RAM: {memory.percent}%")
        
        for gpu in gpus:
            print(f"GPU{gpu.id}: {gpu.load*100:.1f}% | {gpu.memoryUsed}MB/{gpu.memoryTotal}MB")
        
        print("-" * 50)
        time.sleep(5)

if __name__ == "__main__":
    monitor_performance()
```

### 2. Benchmarking

#### Throughput Benchmark
```python
# save as benchmark_throughput.py
import time
import asyncio
import aiohttp
import statistics

async def benchmark_throughput(num_requests=100, concurrent=10):
    """Benchmark API throughput"""
    
    async def make_request(session, prompt):
        start_time = time.time()
        payload = {
            "model": "qwen3-30b",
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        async with session.post("http://localhost:8000/v1/completions", json=payload) as response:
            result = await response.json()
            end_time = time.time()
            
            if response.status == 200:
                tokens = len(result['choices'][0]['text'].split())
                return {
                    'latency': end_time - start_time,
                    'tokens': tokens,
                    'tokens_per_second': tokens / (end_time - start_time)
                }
            else:
                return None
    
    # Test prompts
    prompts = [
        "Explain artificial intelligence in simple terms.",
        "Write a Python function to sort a list.",
        "Describe the benefits of renewable energy.",
        "What is quantum computing?",
        "How does machine learning work?"
    ] * (num_requests // 5 + 1)
    
    async with aiohttp.ClientSession() as session:
        # Warm up
        await make_request(session, "Hello")
        
        # Benchmark
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrent)
        
        async def limited_request(prompt):
            async with semaphore:
                return await make_request(session, prompt)
        
        tasks = [limited_request(prompt) for prompt in prompts[:num_requests]]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # Calculate metrics
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            latencies = [r['latency'] for r in valid_results]
            tokens_per_sec = [r['tokens_per_second'] for r in valid_results]
            
            print(f"Benchmark Results ({len(valid_results)}/{num_requests} successful)")
            print(f"Total time: {end_time - start_time:.2f}s")
            print(f"Requests/sec: {len(valid_results) / (end_time - start_time):.2f}")
            print(f"Average latency: {statistics.mean(latencies):.2f}s")
            print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.2f}s")
            print(f"Average tokens/sec: {statistics.mean(tokens_per_sec):.2f}")

if __name__ == "__main__":
    asyncio.run(benchmark_throughput())
```

## 🎛️ Environment Variables

### Performance Environment Variables
```bash
# CUDA optimizations
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# vLLM optimizations
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_LOGGING_LEVEL=WARNING  # Reduce logging overhead

# Tokenizer optimizations
export TOKENIZERS_PARALLELISM=false

# HuggingFace optimizations
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

# Memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 📈 Scaling Strategies

### Horizontal Scaling
```yaml
# Multiple server instances
# Instance 1: GPUs 0-3
hardware:
  cuda_visible_devices: "0,1,2,3"
server:
  port: 8000

# Instance 2: GPUs 4-7
hardware:
  cuda_visible_devices: "4,5,6,7"
server:
  port: 8001
```

### Load Balancing
```nginx
# nginx.conf for load balancing
upstream qwen3_backend {
    server localhost:8000;
    server localhost:8001;
}

server {
    listen 80;
    location / {
        proxy_pass http://qwen3_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔍 Troubleshooting Performance Issues

### Common Performance Problems

1. **Low GPU Utilization**
   - Increase batch size
   - Enable chunked prefill
   - Check for CPU bottlenecks

2. **High Memory Usage**
   - Reduce `gpu_memory_utilization`
   - Use FP8 KV cache
   - Enable CPU offloading

3. **High Latency**
   - Reduce batch size
   - Optimize tensor parallel size
   - Check network latency

4. **Memory Fragmentation**
   - Restart server periodically
   - Use memory pooling
   - Optimize allocation patterns

### Performance Debugging
```bash
# Profile GPU usage
nsys profile -o profile.qdrep python deploy_qwen3.py

# Monitor memory allocation
python -m torch.utils.bottleneck deploy_qwen3.py

# Check for memory leaks
valgrind --tool=memcheck python deploy_qwen3.py
```

This guide should help you achieve optimal performance for your Qwen3-30B deployment. Remember to test changes incrementally and monitor the impact on both performance and stability.
