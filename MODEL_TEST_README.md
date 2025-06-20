# Qwen3-30B-A3B 离线模型测试套件

这是一个用于测试Qwen3-30B-A3B模型性能的完整测试套件，支持离线推理和性能评估。

## 功能特性

- ✅ **离线推理**: 无需启动API服务器，直接加载模型进行测试
- ✅ **多GPU支持**: 支持8-way tensor parallelism分布式推理
- ✅ **全面测试**: 包含8种不同类型的测试用例
- ✅ **性能监控**: 实时监控GPU内存使用和推理性能
- ✅ **批量测试**: 支持批量推理性能评估
- ✅ **结果保存**: 自动保存详细的测试结果到JSON文件

## 文件说明

### 核心脚本
- `offline_model_test.py` - 完整的性能测试脚本
- `quick_model_test.py` - 快速验证脚本
- `run_tests.sh` - 测试运行脚本

### 配置文件
- `config/model_config.yaml` - 模型配置文件
- `deploy_qwen3.py` - vLLM服务器部署脚本

### 结果文件
- `qwen3_test_results_*.json` - 测试结果文件
- `performance_report.md` - 性能分析报告

## 系统要求

### 硬件要求
- **GPU**: 8 × NVIDIA GeForce RTX 3090 (24GB) 或同等配置
- **内存**: 至少64GB系统内存
- **存储**: 至少100GB可用空间用于模型文件

### 软件要求
- **Python**: 3.10+
- **CUDA**: 12.0+
- **依赖包**:
  ```bash
  pip install torch vllm transformers huggingface-hub GPUtil psutil numpy
  ```

## 快速开始

### 1. 环境检查
```bash
# 检查GPU状态
nvidia-smi

# 检查Python环境
python3 --version

# 检查依赖包
python3 -c "import torch, vllm, GPUtil, psutil"
```

### 2. 模型下载
模型文件应位于以下路径：
```
./models/models--Qwen--Qwen3-30B-A3B/snapshots/ae659febe817e4b3ebd7355f47792725801204c9/
```

如果模型未下载，脚本会自动提示错误。

### 3. 运行测试

#### 方法1: 使用运行脚本（推荐）
```bash
chmod +x run_tests.sh
./run_tests.sh
```

#### 方法2: 直接运行Python脚本
```bash
# 快速测试（验证模型是否正常）
python3 quick_model_test.py

# 完整性能测试
python3 offline_model_test.py
```

## 测试用例

### 单个测试用例
1. **文本生成** - 关于AI发展历程的短文
2. **问答** - 深度学习相关问题
3. **代码生成** - Python快速排序算法
4. **数学推理** - 圆的面积和周长计算
5. **逻辑推理** - 三个盒子的逻辑题
6. **创意写作** - 时间旅行者日记
7. **翻译** - 英文到中文翻译
8. **总结** - 机器学习内容总结

### 批量测试
- 同时处理4个不同类型的请求
- 评估并发处理性能

## 性能指标

### 监控指标
- **响应时间**: 总生成时间
- **吞吐量**: tokens/秒
- **GPU内存使用**: 实时监控8个GPU的内存使用
- **输入/输出tokens**: 统计token数量

### 输出格式
测试结果保存为JSON格式，包含：
```json
{
  "model_path": "模型路径",
  "tensor_parallel_size": 8,
  "test_time": "测试时间",
  "system_info": {
    "cpu_count": 64,
    "memory_total": 811243986944,
    "gpu_count": 8
  },
  "test_results": [...]
}
```

## 配置说明

### 内存配置
```python
gpu_memory_utilization=0.5  # GPU内存使用率
max_model_len=4096         # 最大序列长度
swap_space=2               # 交换空间大小
```

### 推理配置
```python
temperature=0.7            # 采样温度
top_p=0.8                 # Top-p采样
max_tokens=512            # 最大生成tokens
```

## 故障排除

### 常见问题

#### 1. GPU内存不足
```
ValueError: Free memory on device is less than desired GPU memory utilization
```
**解决方案**:
- 降低`gpu_memory_utilization`参数
- 减少`max_model_len`
- 确保没有其他进程占用GPU

#### 2. 模型路径错误
```
错误: 模型路径不存在
```
**解决方案**:
- 检查模型文件是否正确下载
- 验证路径是否正确

#### 3. 依赖包缺失
```
ImportError: No module named 'vllm'
```
**解决方案**:
```bash
pip install vllm transformers huggingface-hub GPUtil psutil
```

### 性能优化建议

1. **提高批量大小**: 批量推理可获得3-4倍性能提升
2. **调整内存使用**: 根据实际需求调整GPU内存利用率
3. **启用编译优化**: 考虑启用CUDA graph优化
4. **网络优化**: 使用NVLink连接的GPU以提高通信效率

## 结果分析

测试完成后，查看生成的文件：
- `qwen3_test_results_*.json` - 详细测试数据
- `performance_report.md` - 性能分析报告

典型性能指标：
- **单个推理**: ~22 tokens/s
- **批量推理**: ~83 tokens/s
- **GPU内存使用**: ~12.6GB per GPU
- **模型加载时间**: ~25秒

## 许可证

本项目遵循MIT许可证。
