# Use official Python 3.11 slim image (small and fast)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first
# (Docker caches this layer — if requirements.txt unchanged, pip install is skipped)
COPY requirements.txt ./requirements.txt

# Install Python dependencies (no cache to keep image small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into /app
COPY . .

# Set Python path so imports like "from models import ..." work from /app
ENV PYTHONPATH=/app

# Port HuggingFace Spaces expects (REQUIRED — do not change)
EXPOSE 7860

# Start the FastAPI server
# "server.app:app" means: the `app` variable in `server/app.py`
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]