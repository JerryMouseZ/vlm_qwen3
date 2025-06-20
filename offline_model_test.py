#!/usr/bin/env python3
"""
Qwen3-30B-A3B 离线模型性能测试脚本

该脚本直接使用vLLM的离线推理功能加载本地模型，
无需启动API服务器，支持多GPU分布式推理和性能评估。
"""

import os
import time
import json
import psutil
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
import GPUtil

try:
    from vllm import LLM, SamplingParams
    from vllm.distributed import init_distributed_environment
except ImportError:
    print("错误: 请安装vLLM库")
    print("pip install vllm")
    exit(1)


@dataclass
class TestResult:
    """测试结果数据类"""
    prompt: str
    response: str
    first_token_time: float
    total_time: float
    input_tokens: int
    output_tokens: int
    throughput: float
    gpu_memory_used: List[float]
    test_type: str


class ModelTester:
    """模型测试器类"""
    
    def __init__(self, model_path: str, tensor_parallel_size: int = 8):
        """
        初始化模型测试器
        
        Args:
            model_path: 本地模型路径
            tensor_parallel_size: 张量并行大小（GPU数量）
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = None
        self.test_results = []
        
        # 测试用例定义
        self.test_cases = self._define_test_cases()
        
    def _define_test_cases(self) -> List[Dict[str, Any]]:
        """定义测试用例"""
        return [
            {
                "type": "文本生成",
                "prompt": "请写一篇关于人工智能发展历程的短文，包括关键里程碑和未来展望。",
                "max_tokens": 512
            },
            {
                "type": "问答",
                "prompt": "什么是深度学习？请详细解释其工作原理和主要应用领域。",
                "max_tokens": 400
            },
            {
                "type": "代码生成",
                "prompt": "请用Python编写一个快速排序算法的实现，包含详细注释。",
                "max_tokens": 600
            },
            {
                "type": "数学推理",
                "prompt": "如果一个圆的半径是5厘米，请计算其面积和周长，并解释计算过程。",
                "max_tokens": 300
            },
            {
                "type": "逻辑推理",
                "prompt": "有三个盒子，一个装金子，一个装银子，一个是空的。每个盒子上都有标签，但所有标签都是错的。如果我从标着'金子'的盒子里拿出一个银币，那么金子在哪个盒子里？",
                "max_tokens": 400
            },
            {
                "type": "创意写作",
                "prompt": "以'时间旅行者的日记'为题，写一个科幻短故事的开头。",
                "max_tokens": 500
            },
            {
                "type": "翻译",
                "prompt": "请将以下英文翻译成中文：'Artificial intelligence is revolutionizing the way we work, learn, and interact with technology.'",
                "max_tokens": 200
            },
            {
                "type": "总结",
                "prompt": "请总结以下内容的要点：机器学习是人工智能的一个分支，它使计算机能够在没有明确编程的情况下学习。机器学习算法通过分析数据来识别模式，并使用这些模式来做出预测或决策。",
                "max_tokens": 250
            }
        ]
    
    def load_model(self):
        """加载模型"""
        print(f"正在加载模型: {self.model_path}")
        print(f"使用 {self.tensor_parallel_size} 个GPU进行张量并行")
        
        start_time = time.time()
        
        try:
            # 配置vLLM参数
            self.llm = LLM(
                model=self.model_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=0.5,  # 降低内存使用
                max_model_len=4096,  # 减少最大序列长度
                trust_remote_code=True,
                enforce_eager=True,  # 使用eager模式减少内存
                swap_space=2,  # 减少swap空间
                block_size=16
            )
            
            load_time = time.time() - start_time
            print(f"模型加载完成，耗时: {load_time:.2f}秒")
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def get_gpu_memory_usage(self) -> List[float]:
        """获取GPU内存使用情况"""
        try:
            gpus = GPUtil.getGPUs()
            return [gpu.memoryUsed for gpu in gpus]
        except:
            return []
    
    def run_single_test(self, test_case: Dict[str, Any]) -> TestResult:
        """运行单个测试用例"""
        prompt = test_case["prompt"]
        max_tokens = test_case["max_tokens"]
        test_type = test_case["type"]
        
        print(f"\n运行测试: {test_type}")
        print(f"Prompt: {prompt[:100]}...")
        
        # 配置采样参数
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            max_tokens=max_tokens,
            stop=None
        )
        
        # 记录开始时间和GPU内存
        start_time = time.time()
        gpu_memory_before = self.get_gpu_memory_usage()
        
        # 执行推理
        outputs = self.llm.generate([prompt], sampling_params)
        
        # 记录结束时间
        end_time = time.time()
        total_time = end_time - start_time
        
        # 获取结果
        output = outputs[0]
        response = output.outputs[0].text
        
        # 计算token数量
        input_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        
        # 计算吞吐量
        throughput = output_tokens / total_time if total_time > 0 else 0
        
        # 获取GPU内存使用
        gpu_memory_after = self.get_gpu_memory_usage()
        
        # 创建测试结果
        result = TestResult(
            prompt=prompt,
            response=response,
            first_token_time=0.0,  # vLLM离线模式无法精确测量首token时间
            total_time=total_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            throughput=throughput,
            gpu_memory_used=gpu_memory_after,
            test_type=test_type
        )
        
        print(f"完成 - 输入tokens: {input_tokens}, 输出tokens: {output_tokens}")
        print(f"总时间: {total_time:.2f}s, 吞吐量: {throughput:.2f} tokens/s")
        
        return result
    
    def run_batch_test(self, batch_size: int = 4) -> List[TestResult]:
        """运行批量测试"""
        print(f"\n开始批量测试 (batch_size={batch_size})")
        
        # 准备批量prompts
        batch_prompts = []
        batch_params = []
        
        for i in range(batch_size):
            test_case = self.test_cases[i % len(self.test_cases)]
            batch_prompts.append(test_case["prompt"])
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.8,
                max_tokens=test_case["max_tokens"],
                stop=None
            )
            batch_params.append(sampling_params)
        
        # 执行批量推理
        start_time = time.time()
        outputs = self.llm.generate(batch_prompts, batch_params[0])  # 使用第一个参数
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # 处理结果
        results = []
        total_output_tokens = 0
        
        for i, output in enumerate(outputs):
            test_case = self.test_cases[i % len(self.test_cases)]
            response = output.outputs[0].text
            input_tokens = len(output.prompt_token_ids)
            output_tokens = len(output.outputs[0].token_ids)
            total_output_tokens += output_tokens
            
            result = TestResult(
                prompt=batch_prompts[i],
                response=response,
                first_token_time=0.0,
                total_time=total_time / batch_size,  # 平均时间
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                throughput=output_tokens / (total_time / batch_size),
                gpu_memory_used=self.get_gpu_memory_usage(),
                test_type=f"批量_{test_case['type']}"
            )
            results.append(result)
        
        batch_throughput = total_output_tokens / total_time
        print(f"批量测试完成 - 总时间: {total_time:.2f}s")
        print(f"批量吞吐量: {batch_throughput:.2f} tokens/s")
        
        return results

    def run_all_tests(self):
        """运行所有测试"""
        print("=" * 60)
        print("开始Qwen3-30B-A3B模型性能测试")
        print("=" * 60)

        # 运行单个测试
        print("\n1. 单个测试用例")
        for test_case in self.test_cases:
            try:
                result = self.run_single_test(test_case)
                self.test_results.append(result)
            except Exception as e:
                print(f"测试失败 ({test_case['type']}): {e}")

        # 运行批量测试
        print("\n2. 批量测试")
        try:
            batch_results = self.run_batch_test(batch_size=4)
            self.test_results.extend(batch_results)
        except Exception as e:
            print(f"批量测试失败: {e}")

    def save_results(self, output_file: str = None):
        """保存测试结果"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"qwen3_test_results_{timestamp}.json"

        # 准备结果数据
        results_data = {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "test_time": datetime.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "gpu_count": len(GPUtil.getGPUs()) if GPUtil.getGPUs() else 0
            },
            "test_results": []
        }

        # 转换测试结果
        for result in self.test_results:
            result_dict = {
                "test_type": result.test_type,
                "prompt": result.prompt[:200] + "..." if len(result.prompt) > 200 else result.prompt,
                "response": result.response[:500] + "..." if len(result.response) > 500 else result.response,
                "metrics": {
                    "first_token_time": result.first_token_time,
                    "total_time": result.total_time,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "throughput": result.throughput,
                    "gpu_memory_used": result.gpu_memory_used
                }
            }
            results_data["test_results"].append(result_dict)

        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)

        print(f"\n测试结果已保存到: {output_file}")

    def print_summary(self):
        """打印测试摘要"""
        if not self.test_results:
            print("没有测试结果")
            return

        print("\n" + "=" * 60)
        print("测试结果摘要")
        print("=" * 60)

        # 按测试类型分组统计
        type_stats = {}
        total_tokens = 0
        total_time = 0

        for result in self.test_results:
            test_type = result.test_type
            if test_type not in type_stats:
                type_stats[test_type] = {
                    "count": 0,
                    "total_time": 0,
                    "total_tokens": 0,
                    "throughputs": []
                }

            type_stats[test_type]["count"] += 1
            type_stats[test_type]["total_time"] += result.total_time
            type_stats[test_type]["total_tokens"] += result.output_tokens
            type_stats[test_type]["throughputs"].append(result.throughput)

            total_tokens += result.output_tokens
            total_time += result.total_time

        # 打印统计信息
        print(f"总测试数量: {len(self.test_results)}")
        print(f"总生成tokens: {total_tokens}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均吞吐量: {total_tokens/total_time:.2f} tokens/秒")

        print("\n按测试类型统计:")
        print("-" * 80)
        print(f"{'测试类型':<15} {'数量':<6} {'平均时间(s)':<12} {'平均吞吐量':<15} {'tokens总数':<10}")
        print("-" * 80)

        for test_type, stats in type_stats.items():
            avg_time = stats["total_time"] / stats["count"]
            avg_throughput = np.mean(stats["throughputs"])

            print(f"{test_type:<15} {stats['count']:<6} {avg_time:<12.2f} "
                  f"{avg_throughput:<15.2f} {stats['total_tokens']:<10}")

        # GPU内存使用情况
        if self.test_results and self.test_results[0].gpu_memory_used:
            print(f"\nGPU内存使用情况:")
            latest_memory = self.test_results[-1].gpu_memory_used
            for i, memory in enumerate(latest_memory):
                print(f"GPU {i}: {memory:.1f} MB")


def main():
    """主函数"""
    # 模型路径
    model_path = "./models/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9"

    # 检查模型路径是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        print("请确保已下载模型到指定路径")
        return

    # 检查GPU数量
    try:
        gpu_count = torch.cuda.device_count()
        print(f"检测到 {gpu_count} 个GPU")

        if gpu_count == 0:
            print("错误: 未检测到GPU")
            return

        # 使用所有可用GPU，最多8个
        tensor_parallel_size = min(gpu_count, 8)

    except Exception as e:
        print(f"GPU检测失败: {e}")
        return

    try:
        # 创建测试器
        tester = ModelTester(model_path, tensor_parallel_size)

        # 加载模型
        tester.load_model()

        # 运行测试
        tester.run_all_tests()

        # 打印摘要
        tester.print_summary()

        # 保存结果
        tester.save_results()

        print("\n测试完成!")

    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
