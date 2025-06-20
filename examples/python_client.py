#!/usr/bin/env python3
"""
Python client examples for Qwen3-30B vLLM API
Demonstrates various ways to interact with the deployed model.
"""

import openai
import requests
import json
import asyncio
import aiohttp
from typing import List, Dict, Any

class Qwen3Client:
    def __init__(self, base_url: str = "http://localhost:8000", model_name: str = "qwen3-30b"):
        """Initialize the Qwen3 client."""
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        
        # Initialize OpenAI client for compatibility
        self.openai_client = openai.OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=f"{base_url}/v1"
        )
    
    def simple_completion(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate a simple text completion."""
        try:
            response = self.openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error in simple completion: {e}")
            return ""
    
    def chat_completion(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.7) -> str:
        """Generate a chat completion."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in chat completion: {e}")
            return ""
    
    def streaming_chat(self, messages: List[Dict[str, str]], max_tokens: int = 200, temperature: float = 0.7):
        """Generate a streaming chat completion."""
        try:
            stream = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error in streaming chat: {e}")
    
    def code_generation(self, prompt: str, language: str = "python") -> str:
        """Generate code with optimized parameters."""
        code_prompt = f"Write a {language} function for the following task:\n{prompt}\n\n```{language}\n"
        
        try:
            response = self.openai_client.completions.create(
                model=self.model_name,
                prompt=code_prompt,
                max_tokens=300,
                temperature=0.2,  # Lower temperature for more deterministic code
                top_p=0.9,
                stop=["```", "\n\n\n"]
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error in code generation: {e}")
            return ""
    
    def question_answering(self, question: str, context: str = "") -> str:
        """Answer questions with optional context."""
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        else:
            prompt = f"Question: {question}\nAnswer:"
        
        try:
            response = self.openai_client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=200,
                temperature=0.3,
                top_p=0.9
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error in question answering: {e}")
            return ""
    
    def creative_writing(self, prompt: str, style: str = "narrative") -> str:
        """Generate creative content with higher temperature."""
        creative_prompt = f"Write a {style} based on this prompt: {prompt}\n\n"
        
        try:
            response = self.openai_client.completions.create(
                model=self.model_name,
                prompt=creative_prompt,
                max_tokens=400,
                temperature=0.9,  # Higher temperature for creativity
                top_p=0.95,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(f"Error in creative writing: {e}")
            return ""
    
    async def batch_completions(self, prompts: List[str], max_tokens: int = 100) -> List[str]:
        """Process multiple prompts concurrently."""
        async def process_prompt(session, prompt):
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            async with session.post(f"{self.base_url}/v1/completions", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result['choices'][0]['text'].strip()
                else:
                    return f"Error: {response.status}"
        
        async with aiohttp.ClientSession() as session:
            tasks = [process_prompt(session, prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            return results

def demonstrate_features():
    """Demonstrate various features of the Qwen3 client."""
    print("=== Qwen3-30B Python Client Examples ===\n")
    
    client = Qwen3Client()
    
    # 1. Simple completion
    print("1. Simple Text Completion:")
    result = client.simple_completion("The benefits of renewable energy include")
    print(f"Result: {result}\n")
    
    # 2. Chat completion
    print("2. Chat Completion:")
    messages = [
        {"role": "user", "content": "What are the main principles of machine learning?"}
    ]
    result = client.chat_completion(messages)
    print(f"Result: {result}\n")
    
    # 3. Multi-turn conversation
    print("3. Multi-turn Conversation:")
    conversation = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a high-level programming language known for its simplicity and readability."},
        {"role": "user", "content": "What makes it popular for data science?"}
    ]
    result = client.chat_completion(conversation)
    print(f"Result: {result}\n")
    
    # 4. Code generation
    print("4. Code Generation:")
    result = client.code_generation("Calculate the factorial of a number using recursion")
    print(f"Generated code:\n{result}\n")
    
    # 5. Question answering
    print("5. Question Answering:")
    context = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
    question = "What is machine learning?"
    result = client.question_answering(question, context)
    print(f"Answer: {result}\n")
    
    # 6. Creative writing
    print("6. Creative Writing:")
    result = client.creative_writing("A robot discovers emotions for the first time", "short story")
    print(f"Story: {result[:200]}...\n")
    
    # 7. Streaming example
    print("7. Streaming Chat:")
    messages = [{"role": "user", "content": "Explain the concept of neural networks"}]
    print("Streaming response: ", end="", flush=True)
    for chunk in client.streaming_chat(messages):
        print(chunk, end="", flush=True)
    print("\n")
    
    # 8. Batch processing
    print("8. Batch Processing:")
    prompts = [
        "Define artificial intelligence:",
        "What is quantum computing?",
        "Explain blockchain technology:"
    ]
    
    async def run_batch():
        results = await client.batch_completions(prompts)
        for i, (prompt, result) in enumerate(zip(prompts, results), 1):
            print(f"Batch {i}: {prompt}")
            print(f"Result: {result[:100]}...\n")
    
    asyncio.run(run_batch())

def interactive_chat():
    """Interactive chat session with Qwen3."""
    print("=== Interactive Chat with Qwen3-30B ===")
    print("Type 'quit' to exit, 'clear' to clear conversation history\n")
    
    client = Qwen3Client()
    conversation = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'clear':
            conversation = []
            print("Conversation history cleared.\n")
            continue
        elif not user_input:
            continue
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # Get response
        print("Qwen3: ", end="", flush=True)
        response_text = ""
        for chunk in client.streaming_chat(conversation):
            print(chunk, end="", flush=True)
            response_text += chunk
        print("\n")
        
        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": response_text})

def benchmark_performance():
    """Benchmark the performance of the deployed model."""
    print("=== Performance Benchmark ===\n")
    
    client = Qwen3Client()
    
    # Test different prompt lengths
    test_cases = [
        ("Short prompt", "Hello, how are you?"),
        ("Medium prompt", "Explain the concept of artificial intelligence and its applications in modern technology."),
        ("Long prompt", "Write a detailed analysis of the impact of machine learning on various industries including healthcare, finance, transportation, and education. Discuss both the benefits and challenges.")
    ]
    
    for test_name, prompt in test_cases:
        print(f"Testing {test_name}:")
        
        import time
        start_time = time.time()
        result = client.simple_completion(prompt, max_tokens=200)
        end_time = time.time()
        
        duration = end_time - start_time
        tokens = len(result.split())  # Rough token count
        tokens_per_sec = tokens / duration if duration > 0 else 0
        
        print(f"  Duration: {duration:.2f}s")
        print(f"  Tokens: ~{tokens}")
        print(f"  Speed: ~{tokens_per_sec:.1f} tokens/s")
        print(f"  Response: {result[:100]}...\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-30B Python Client Examples")
    parser.add_argument("--mode", choices=["demo", "chat", "benchmark"], default="demo",
                       help="Mode to run: demo (show examples), chat (interactive), benchmark (performance test)")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the vLLM server")
    parser.add_argument("--model", default="qwen3-30b", help="Model name")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        demonstrate_features()
    elif args.mode == "chat":
        interactive_chat()
    elif args.mode == "benchmark":
        benchmark_performance()
