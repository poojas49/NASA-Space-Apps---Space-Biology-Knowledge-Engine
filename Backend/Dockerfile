# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including Streamlit dependencies)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose ports
# Port 5001 for Flask API
# Port 8501 for Streamlit (if needed)
EXPOSE 5001 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the Flask API server
CMD ["python", "api_server.py"]