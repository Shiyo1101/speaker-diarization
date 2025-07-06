ARG UV_VERSION=latest
ARG DEBIAN_VERSION=bookworm


FROM ghcr.io/astral-sh/uv:$UV_VERSION AS uv


FROM mcr.microsoft.com/vscode/devcontainers/base:$DEBIAN_VERSION

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    cmake pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv --chown=vscode: /uv /uvx /bin/
