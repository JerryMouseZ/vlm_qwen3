#!/bin/bash
# Curl examples for testing Qwen3-30B vLLM API
# Make sure the server is running before executing these commands

BASE_URL="http://localhost:8000"
MODEL_NAME="qwen3-30b"

echo "=== Qwen3-30B vLLM API Examples ==="
echo "Base URL: $BASE_URL"
echo "Model: $MODEL_NAME"
echo ""

# 1. Health Check
echo "1. Health Check:"
curl -X GET "$BASE_URL/health" \
  -H "Content-Type: application/json" | jq .
echo -e "\n"

# 2. List Models
echo "2. List Available Models:"
curl -X GET "$BASE_URL/v1/models" \
  -H "Content-Type: application/json" | jq .
echo -e "\n"

# 3. Simple Completion
echo "3. Simple Text Completion:"
curl -X POST "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "prompt": "The future of artificial intelligence is",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }' | jq .
echo -e "\n"

# 4. Chat Completion
echo "4. Chat Completion:"
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }' | jq .
echo -e "\n"

# 5. Multi-turn Chat
echo "5. Multi-turn Chat Conversation:"
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "What is machine learning?"},
      {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
      {"role": "user", "content": "Can you give me a practical example?"}
    ],
    "max_tokens": 150,
    "temperature": 0.8
  }' | jq .
echo -e "\n"

# 6. Code Generation
echo "6. Code Generation:"
curl -X POST "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "prompt": "Write a Python function to calculate the factorial of a number:\n\ndef factorial(n):",
    "max_tokens": 150,
    "temperature": 0.3,
    "stop": ["\n\n", "def "]
  }' | jq .
echo -e "\n"

# 7. Streaming Chat (Note: This will show raw streaming output)
echo "7. Streaming Chat Completion:"
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Write a short poem about technology"}
    ],
    "max_tokens": 100,
    "temperature": 0.8,
    "stream": true
  }'
echo -e "\n\n"

# 8. Creative Writing with Higher Temperature
echo "8. Creative Writing (High Temperature):"
curl -X POST "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "prompt": "Once upon a time in a distant galaxy,",
    "max_tokens": 200,
    "temperature": 1.0,
    "top_p": 0.95,
    "frequency_penalty": 0.5
  }' | jq .
echo -e "\n"

# 9. Technical Explanation (Low Temperature)
echo "9. Technical Explanation (Low Temperature):"
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Explain the difference between TCP and UDP protocols"}
    ],
    "max_tokens": 250,
    "temperature": 0.2,
    "top_p": 0.9
  }' | jq .
echo -e "\n"

# 10. Math Problem Solving
echo "10. Math Problem Solving:"
curl -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'$MODEL_NAME'",
    "messages": [
      {"role": "user", "content": "Solve this step by step: If a train travels 120 km in 2 hours, what is its average speed? Then, how long would it take to travel 300 km at the same speed?"}
    ],
    "max_tokens": 200,
    "temperature": 0.1
  }' | jq .
echo -e "\n"

echo "=== All examples completed ==="
echo "Note: Install 'jq' for better JSON formatting: sudo apt-get install jq"
