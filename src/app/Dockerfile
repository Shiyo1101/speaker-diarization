FROM python:3.9-slim

# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

WORKDIR /app

COPY pyproject.toml ./

RUN uv venv /opt/venv && \
    uv pip install --no-cache-dir -p /opt/venv/bin/python -r pyproject.toml

COPY src/app ./src/app
COPY src/app/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ARG HUGGING_FACE_TOKEN
ENV HUGGING_FACE_TOKEN=${HUGGING_FACE_TOKEN}


ENV PATH="/opt/venv/bin:$PATH"

CMD ["/app/entrypoint.sh"]
