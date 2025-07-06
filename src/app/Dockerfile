FROM python:3.9-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/

WORKDIR /app
COPY pyproject.toml ./

RUN uv venv /opt/venv && \
    uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --no-cache-dir --python /opt/venv/bin/python -r requirements.txt

FROM python:3.9-slim

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY src/ ./src
COPY src/app/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
