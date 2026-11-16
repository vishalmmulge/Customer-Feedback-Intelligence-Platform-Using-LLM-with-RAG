# ---------------------------------------------------------------
# Ollama Dockerfile
#
# Starts ollama serve, waits until ready, then pulls the required
# models exactly once into a persistent volume. Models are NOT baked
# into the image; they are downloaded on first container start and
# persisted via Docker volume.
#
# The sed command strips Windows CRLF line endings from entrypoint.sh
# so the script runs correctly inside the Linux container when the
# project is developed on Windows.
#
# curl is installed to support the Docker health check endpoint.
# ---------------------------------------------------------------
FROM ollama/ollama:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY ollama/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
