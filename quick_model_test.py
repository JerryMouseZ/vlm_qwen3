#!/usr/bin/env python3
"""
Qwen3-30B-A3B 快速模型测试脚本

用于快速验证模型是否正常加载和推理
"""

import os
import time
import torch
from vllm import LLM, SamplingParams


def test_model_basic():
    """基础模型测试"""
    model_path = "./models/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9"
    
    # 检查模型路径
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    # 检查GPU
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU")
    
    if gpu_count == 0:
        print("错误: 未检测到GPU")
        return False
    
    tensor_parallel_size = min(gpu_count, 8)
    print(f"使用 {tensor_parallel_size} 个GPU进行推理")
    
    try:
        print("正在加载模型...")
        start_time = time.time()
        
        # 加载模型
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.5,  # 进一步降低内存使用
            max_model_len=2048,  # 减少内存使用
            trust_remote_code=True,
            enforce_eager=True,  # 使用eager模式减少内存
            swap_space=2  # 减少swap空间
        )
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f}秒")
        
        # 简单测试
        test_prompts = [
            "你好，请介绍一下你自己。",
            "什么是人工智能？",
            "请写一个Python的Hello World程序。"
        ]
        
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=200
        )
        
        print("\n开始推理测试...")
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n测试 {i}: {prompt}")
            
            start_time = time.time()
            outputs = llm.generate([prompt], sampling_params)
            inference_time = time.time() - start_time
            
            response = outputs[0].outputs[0].text
            output_tokens = len(outputs[0].outputs[0].token_ids)
            throughput = output_tokens / inference_time
            
            print(f"响应: {response[:100]}...")
            print(f"生成tokens: {output_tokens}, 耗时: {inference_time:.2f}s, 吞吐量: {throughput:.2f} tokens/s")
        
        print("\n✅ 模型测试成功!")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("Qwen3-30B-A3B 快速模型测试")
    print("=" * 50)
    
    success = test_model_basic()
    
    if success:
        print("\n🎉 模型工作正常，可以运行完整测试脚本!")
    else:
        print("\n⚠️  模型测试失败，请检查配置和环境")
