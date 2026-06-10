# Use the updated official CPU-focused server image from the ggml-org registry
FROM ghcr.io/ggml-org/llama.cpp:server

# Tell the container's dynamic linker to look inside /app for shared libraries
ENV LD_LIBRARY_PATH=/app

# --- DYNAMIC CONFIGURATION DEFAULTS ---
ENV LLAMA_THREADS=4
ENV LLAMA_CTX=2048
ENV LLAMA_HOST=0.0.0.0
ENV LLAMA_PORT=8080
ENV LLAMA_MODEL=/locallm/model.gguf

# Create a directory inside the container for the model
WORKDIR /locallm

# Copy the GGUF file from your host into the container image
# COPY ./models/qwen3-4b.gguf /locallm/model.gguf

# Expose the server port
EXPOSE 8080

# FIX: Drop ENTRYPOINT entirely, and let CMD handle the shell parsing cleanly
ENTRYPOINT []
CMD ["sh", "-c", "/app/llama-server -m ${LLAMA_MODEL} --host ${LLAMA_HOST} --port ${LLAMA_PORT} -t ${LLAMA_THREADS} -c ${LLAMA_CTX}"]

