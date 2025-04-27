# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system packages needed to compile Python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default command to run your app
CMD ["python", "your_app.py"]
