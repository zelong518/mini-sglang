curl http://127.0.0.1:1919/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/volume/posttrain/models/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'