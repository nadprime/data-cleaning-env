FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (cached layer)
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy all project files
COPY . .

ENV PYTHONPATH=/app

EXPOSE 7860

# Start server via uv — exactly as judges will test
CMD ["uv", "run", "server"]