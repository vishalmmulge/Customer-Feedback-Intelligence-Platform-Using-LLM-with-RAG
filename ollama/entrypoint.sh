#!/bin/bash
# ---------------------------------------------------------------
# Ollama model initialization entrypoint.
#
# Starts the Ollama server, waits until it is ready, then pulls the
# required embedding and LLM models exactly once. Models are stored in
# /root/.ollama which is backed by a Docker volume, so they persist
# across container restarts and are not re-downloaded every time.
# ---------------------------------------------------------------
set -e

MODELS="${OLLAMA_MODELS:-nomic-embed-text llama3.2:3b}"

# Start ollama server in the background.
ollama serve &
OLLAMA_PID=$!

# Wait until Ollama is ready (poll /api_tags).
echo "Waiting for Ollama to become ready..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "Ollama is ready."
        break
    fi
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "Ollama server exited unexpectedly."
        exit 1
    fi
    sleep 2
done

# Pull each model if not already present.
for model in $MODELS; do
    echo "Checking model: $model"
    if ollama list 2>/dev/null | grep -q "^${model}"; then
        echo "Model $model already present, skipping pull."
    else
        echo "Pulling model $model (this may take a while)..."
        ollama pull "$model"
    fi
done

echo "All models available. Ollama initialization complete."

# Keep the container running by waiting on the ollama server process.
wait $OLLAMA_PID
