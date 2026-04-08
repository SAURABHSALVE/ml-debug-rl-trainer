FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
# We use --no-cache-dir to keep the image small and ensure fresh downloads
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set PYTHONPATH to ensure 'ml_env' and 'server' are discoverable
ENV PYTHONPATH=/app

# HF Space standard port
EXPOSE 7860

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]