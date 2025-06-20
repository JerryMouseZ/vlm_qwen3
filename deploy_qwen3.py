#!/usr/bin/env python3
"""
vLLM Deployment Script for Qwen3-30B Model
Supports both A2B and A3B variants with comprehensive error handling and monitoring.
"""

import os
import sys
import yaml
import logging
import argparse
import subprocess
import time
import signal
import json
from pathlib import Path
from typing import Dict, Any, Optional
import requests
import psutil
import GPUtil

class Qwen3Deployer:
    def __init__(self, config_path: str = "config/model_config.yaml"):
        """Initialize the Qwen3 deployer with configuration."""
        self.config_path = config_path
        self.config = self.load_config()
        self.setup_logging()
        self.vllm_process = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"Error: Configuration file {self.config_path} not found!")
            sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing configuration file: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format=self.config['logging']['format'],
            handlers=[
                logging.FileHandler(self.config['logging']['log_file']),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_system_requirements(self) -> bool:
        """Check if system meets requirements for deployment."""
        self.logger.info("Checking system requirements...")
        
        # Check GPU availability
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                self.logger.error("No GPUs found!")
                return False
            
            required_gpus = self.config['model']['tensor_parallel_size']
            if len(gpus) < required_gpus:
                self.logger.error(f"Need {required_gpus} GPUs, but only {len(gpus)} available")
                return False
            
            # Check GPU memory
            min_memory_gb = 20  # Minimum memory per GPU for 30B model
            for i, gpu in enumerate(gpus[:required_gpus]):
                if gpu.memoryTotal < min_memory_gb * 1024:  # Convert to MB
                    self.logger.error(f"GPU {i} has insufficient memory: {gpu.memoryTotal}MB < {min_memory_gb}GB")
                    return False
                    
            self.logger.info(f"Found {len(gpus)} GPUs, using {required_gpus} for deployment")
            
        except Exception as e:
            self.logger.error(f"Error checking GPU requirements: {e}")
            return False
        
        # Check available RAM
        memory = psutil.virtual_memory()
        required_ram_gb = 32  # Minimum RAM for 30B model
        if memory.total < required_ram_gb * 1024**3:
            self.logger.warning(f"Low system RAM: {memory.total / 1024**3:.1f}GB < {required_ram_gb}GB")
        
        # Check disk space
        disk = psutil.disk_usage('/')
        required_disk_gb = 100  # Space for model, cache, logs
        if disk.free < required_disk_gb * 1024**3:
            self.logger.warning(f"Low disk space: {disk.free / 1024**3:.1f}GB < {required_disk_gb}GB")
        
        self.logger.info("System requirements check passed!")
        return True
    
    def setup_environment(self):
        """Setup environment variables for optimal performance."""
        env_vars = self.config.get('environment', {})
        
        # Set CUDA devices
        cuda_devices = self.config['hardware']['cuda_visible_devices']
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        
        # Set PyTorch CUDA allocator configuration
        pytorch_conf = self.config['hardware']['pytorch_cuda_alloc_conf']
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = pytorch_conf
        
        # Set other environment variables
        for key, value in env_vars.items():
            os.environ[key] = str(value)
        
        self.logger.info("Environment variables configured")
    
    def build_vllm_command(self) -> list:
        """Build the vLLM server command with all parameters."""
        model_config = self.config['model']
        server_config = self.config['server']
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_config['name'],
            "--host", server_config['host'],
            "--port", str(server_config['port']),
            "--tensor-parallel-size", str(model_config['tensor_parallel_size']),
            "--pipeline-parallel-size", str(model_config['pipeline_parallel_size']),
            "--gpu-memory-utilization", str(model_config['gpu_memory_utilization']),
            "--max-model-len", str(model_config['max_model_len']),
            "--max-num-batched-tokens", str(model_config['max_num_batched_tokens']),
            "--max-num-seqs", str(model_config['max_num_seqs']),
            "--swap-space", str(model_config['swap_space']),
            "--block-size", str(model_config['block_size']),
            "--served-model-name", server_config['served_model_name'],
        ]
        
        # Add optional parameters
        if model_config.get('trust_remote_code'):
            cmd.append("--trust-remote-code")
        
        if model_config.get('quantization'):
            cmd.extend(["--quantization", model_config['quantization']])
        
        if model_config.get('enable_chunked_prefill'):
            cmd.append("--enable-chunked-prefill")
        
        # Removed unsupported arguments for this vLLM version
        # if model_config.get('attention_backend'):
        #     cmd.extend(["--attention-backend", model_config['attention_backend']])

        if model_config.get('kv_cache_dtype') and model_config.get('kv_cache_dtype') != "auto":
            cmd.extend(["--kv-cache-dtype", model_config['kv_cache_dtype']])

        # if self.config['logging'].get('enable_metrics'):
        #     cmd.extend(["--enable-metrics", "--metrics-port", str(self.config['logging']['metrics_port'])])
        
        return cmd
    
    def start_server(self) -> bool:
        """Start the vLLM server."""
        self.logger.info("Starting vLLM server...")
        
        cmd = self.build_vllm_command()
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # Redirect output to log files
            log_file = open(self.config['logging']['log_file'], 'a')
            
            self.vllm_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy()
            )
            
            self.logger.info(f"vLLM server started with PID: {self.vllm_process.pid}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start vLLM server: {e}")
            return False
    
    def wait_for_server(self, timeout: int = 300) -> bool:
        """Wait for the server to be ready."""
        self.logger.info("Waiting for server to be ready...")
        
        url = f"http://{self.config['server']['host']}:{self.config['server']['port']}/health"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.logger.info("Server is ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if self.vllm_process and self.vllm_process.poll() is not None:
                self.logger.error("vLLM process terminated unexpectedly")
                return False
            
            time.sleep(5)
        
        self.logger.error(f"Server failed to start within {timeout} seconds")
        return False
    
    def test_inference(self) -> bool:
        """Test the deployed model with a simple inference request."""
        self.logger.info("Testing model inference...")
        
        url = f"http://{self.config['server']['host']}:{self.config['server']['port']}/v1/completions"
        
        test_data = {
            "model": self.config['server']['served_model_name'],
            "prompt": "Hello, how are you?",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, json=test_data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                self.logger.info("Inference test successful!")
                self.logger.info(f"Response: {result['choices'][0]['text'][:100]}...")
                return True
            else:
                self.logger.error(f"Inference test failed: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Inference test error: {e}")
            return False
    
    def stop_server(self):
        """Stop the vLLM server gracefully."""
        if self.vllm_process:
            self.logger.info("Stopping vLLM server...")
            self.vllm_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.vllm_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.logger.warning("Force killing vLLM server...")
                self.vllm_process.kill()
            
            self.logger.info("vLLM server stopped")
    
    def deploy(self) -> bool:
        """Main deployment function."""
        self.logger.info("Starting Qwen3-30B deployment...")
        
        # Check system requirements
        if not self.check_system_requirements():
            return False
        
        # Setup environment
        self.setup_environment()
        
        # Start server
        if not self.start_server():
            return False
        
        # Wait for server to be ready
        if not self.wait_for_server():
            self.stop_server()
            return False
        
        # Test inference
        if not self.test_inference():
            self.logger.warning("Inference test failed, but server is running")
        
        self.logger.info("Deployment completed successfully!")
        self.logger.info(f"Server running at: http://{self.config['server']['host']}:{self.config['server']['port']}")
        self.logger.info(f"API documentation: http://{self.config['server']['host']}:{self.config['server']['port']}/docs")
        
        return True

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print("\nReceived shutdown signal. Stopping server...")
    if hasattr(signal_handler, 'deployer'):
        signal_handler.deployer.stop_server()
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Deploy Qwen3-30B model with vLLM")
    parser.add_argument("--config", default="config/model_config.yaml", help="Path to configuration file")
    parser.add_argument("--model-variant", choices=["A2B", "A3B"], default="A3B", help="Model variant to deploy")
    args = parser.parse_args()
    
    # Create deployer instance
    deployer = Qwen3Deployer(args.config)
    
    # Update model name based on variant
    if args.model_variant == "A2B":
        deployer.config['model']['name'] = "Qwen/Qwen3-30B-A2B"
    else:
        deployer.config['model']['name'] = "Qwen/Qwen3-30B-A3B"
    
    # Setup signal handlers
    signal_handler.deployer = deployer
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Deploy the model
    success = deployer.deploy()
    
    if success:
        print("Deployment successful! Press Ctrl+C to stop the server.")
        try:
            # Keep the script running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        print("Deployment failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
