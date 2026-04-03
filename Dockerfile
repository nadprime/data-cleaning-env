FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Remove any existing venv and create fresh one
RUN rm -rf .venv && \
    uv venv --python python3.11 && \
    uv sync --frozen --no-dev

# Copy all project files
COPY . .

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

CMD ["uv", "run", "server"]