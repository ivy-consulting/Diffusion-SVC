# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file to the container
COPY requirements.txt .

# Update system packages and install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    python3-dev \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the desired port
EXPOSE 8000

# Define the entry point
CMD ["uvicorn", "server_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
