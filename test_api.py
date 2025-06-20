#!/usr/bin/env python3
"""
Test script for Qwen3-30B vLLM API deployment
Provides comprehensive testing of the deployed model endpoints.
"""

import requests
import json
import time
import argparse
from typing import Dict, Any, List
import asyncio
import aiohttp

class Qwen3APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API tester."""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
    def test_health(self) -> bool:
        """Test the health endpoint."""
        print("Testing health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("✓ Health check passed")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False
    
    def test_models_endpoint(self) -> bool:
        """Test the models listing endpoint."""
        print("\nTesting models endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/v1/models", timeout=10)
            if response.status_code == 200:
                models = response.json()
                print("✓ Models endpoint working")
                print(f"Available models: {[model['id'] for model in models['data']]}")
                return True
            else:
                print(f"✗ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Models endpoint error: {e}")
            return False
    
    def test_completions(self, model_name: str = "qwen3-30b") -> bool:
        """Test the completions endpoint."""
        print(f"\nTesting completions endpoint with model: {model_name}...")
        
        test_prompts = [
            "Hello, how are you today?",
            "Explain quantum computing in simple terms:",
            "Write a Python function to calculate fibonacci numbers:",
            "What is the capital of France?",
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt[:50]}...")
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": 150,
                "temperature": 0.7,
                "top_p": 0.9,
                "stop": ["<|endoftext|>", "<|im_end|>"]
            }
            
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/v1/completions",
                    json=payload,
                    timeout=60
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    completion = result['choices'][0]['text']
                    tokens = result['usage']['completion_tokens']
                    duration = end_time - start_time
                    tokens_per_sec = tokens / duration if duration > 0 else 0
                    
                    print(f"✓ Response received in {duration:.2f}s ({tokens_per_sec:.1f} tokens/s)")
                    print(f"Generated {tokens} tokens")
                    print(f"Response: {completion[:100]}...")
                else:
                    print(f"✗ Request failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"✗ Request error: {e}")
                return False
        
        return True
    
    def test_chat_completions(self, model_name: str = "qwen3-30b") -> bool:
        """Test the chat completions endpoint."""
        print(f"\nTesting chat completions endpoint with model: {model_name}...")
        
        test_conversations = [
            [
                {"role": "user", "content": "Hello! Can you help me with a coding question?"},
            ],
            [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
                {"role": "user", "content": "Can you give me a simple example?"},
            ],
        ]
        
        for i, messages in enumerate(test_conversations, 1):
            print(f"\nChat test {i}...")
            
            payload = {
                "model": model_name,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
            
            try:
                start_time = time.time()
                response = self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=60
                )
                end_time = time.time()
                
                if response.status_code == 200:
                    result = response.json()
                    message = result['choices'][0]['message']['content']
                    tokens = result['usage']['completion_tokens']
                    duration = end_time - start_time
                    tokens_per_sec = tokens / duration if duration > 0 else 0
                    
                    print(f"✓ Chat response received in {duration:.2f}s ({tokens_per_sec:.1f} tokens/s)")
                    print(f"Generated {tokens} tokens")
                    print(f"Response: {message[:100]}...")
                else:
                    print(f"✗ Chat request failed: {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"✗ Chat request error: {e}")
                return False
        
        return True
    
    def test_streaming(self, model_name: str = "qwen3-30b") -> bool:
        """Test streaming completions."""
        print(f"\nTesting streaming completions with model: {model_name}...")
        
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": "Write a short story about a robot learning to paint."}
            ],
            "max_tokens": 300,
            "temperature": 0.8,
            "stream": True
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                print("✓ Streaming started...")
                tokens_received = 0
                start_time = time.time()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]  # Remove 'data: ' prefix
                            if data.strip() == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                if 'choices' in chunk and chunk['choices']:
                                    delta = chunk['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        tokens_received += 1
                                        if tokens_received <= 10:  # Show first few tokens
                                            print(f"Token {tokens_received}: {delta['content']}", end='', flush=True)
                            except json.JSONDecodeError:
                                continue
                
                end_time = time.time()
                duration = end_time - start_time
                tokens_per_sec = tokens_received / duration if duration > 0 else 0
                
                print(f"\n✓ Streaming completed: {tokens_received} tokens in {duration:.2f}s ({tokens_per_sec:.1f} tokens/s)")
                return True
            else:
                print(f"✗ Streaming failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"✗ Streaming error: {e}")
            return False
    
    def test_performance(self, model_name: str = "qwen3-30b", num_requests: int = 5) -> Dict[str, float]:
        """Test performance with multiple concurrent requests."""
        print(f"\nTesting performance with {num_requests} concurrent requests...")
        
        async def make_request(session, prompt_id):
            payload = {
                "model": model_name,
                "prompt": f"Test prompt {prompt_id}: Explain the concept of artificial intelligence.",
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            start_time = time.time()
            async with session.post(f"{self.base_url}/v1/completions", json=payload) as response:
                result = await response.json()
                end_time = time.time()
                
                return {
                    "duration": end_time - start_time,
                    "tokens": result['usage']['completion_tokens'],
                    "status": response.status
                }
        
        async def run_performance_test():
            async with aiohttp.ClientSession() as session:
                tasks = [make_request(session, i) for i in range(num_requests)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                return results
        
        try:
            results = asyncio.run(run_performance_test())
            
            successful_results = [r for r in results if isinstance(r, dict) and r['status'] == 200]
            
            if successful_results:
                avg_duration = sum(r['duration'] for r in successful_results) / len(successful_results)
                total_tokens = sum(r['tokens'] for r in successful_results)
                total_time = max(r['duration'] for r in successful_results)
                throughput = total_tokens / total_time if total_time > 0 else 0
                
                print(f"✓ Performance test completed:")
                print(f"  Successful requests: {len(successful_results)}/{num_requests}")
                print(f"  Average latency: {avg_duration:.2f}s")
                print(f"  Total throughput: {throughput:.1f} tokens/s")
                
                return {
                    "avg_latency": avg_duration,
                    "throughput": throughput,
                    "success_rate": len(successful_results) / num_requests
                }
            else:
                print("✗ All performance test requests failed")
                return {}
                
        except Exception as e:
            print(f"✗ Performance test error: {e}")
            return {}
    
    def run_all_tests(self, model_name: str = "qwen3-30b") -> bool:
        """Run all tests."""
        print("=" * 60)
        print("Qwen3-30B vLLM API Test Suite")
        print("=" * 60)
        
        tests = [
            ("Health Check", lambda: self.test_health()),
            ("Models Endpoint", lambda: self.test_models_endpoint()),
            ("Completions", lambda: self.test_completions(model_name)),
            ("Chat Completions", lambda: self.test_chat_completions(model_name)),
            ("Streaming", lambda: self.test_streaming(model_name)),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                results.append(result)
            except Exception as e:
                print(f"✗ {test_name} failed with exception: {e}")
                results.append(False)
        
        # Performance test
        print(f"\n{'='*20} Performance Test {'='*20}")
        perf_results = self.test_performance(model_name)
        
        # Summary
        print(f"\n{'='*20} Test Summary {'='*20}")
        passed = sum(1 for r in results if r)
        total = len(results)
        print(f"Tests passed: {passed}/{total}")
        
        if perf_results:
            print(f"Average latency: {perf_results.get('avg_latency', 0):.2f}s")
            print(f"Throughput: {perf_results.get('throughput', 0):.1f} tokens/s")
            print(f"Success rate: {perf_results.get('success_rate', 0)*100:.1f}%")
        
        return passed == total

def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-30B vLLM API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the vLLM server")
    parser.add_argument("--model", default="qwen3-30b", help="Model name to test")
    parser.add_argument("--test", choices=["health", "models", "completions", "chat", "streaming", "performance", "all"], 
                       default="all", help="Specific test to run")
    args = parser.parse_args()
    
    tester = Qwen3APITester(args.url)
    
    if args.test == "all":
        success = tester.run_all_tests(args.model)
    elif args.test == "health":
        success = tester.test_health()
    elif args.test == "models":
        success = tester.test_models_endpoint()
    elif args.test == "completions":
        success = tester.test_completions(args.model)
    elif args.test == "chat":
        success = tester.test_chat_completions(args.model)
    elif args.test == "streaming":
        success = tester.test_streaming(args.model)
    elif args.test == "performance":
        results = tester.test_performance(args.model)
        success = bool(results)
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)

if __name__ == "__main__":
    main()
