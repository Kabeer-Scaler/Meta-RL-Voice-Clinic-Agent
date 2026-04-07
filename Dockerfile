FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY app.py .
COPY scenarios/ ./scenarios/

# Expose port 7860 (Hugging Face Spaces standard)
EXPOSE 7860

# Start uvicorn server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
