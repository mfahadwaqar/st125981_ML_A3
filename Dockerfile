# Use Python 3.12 base image
FROM python:3.12.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for scientific packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the port used by Dash
EXPOSE 80

# Run the Dash app
CMD ["python", "-m", "pytest", "test_model.py", "-v"]


# FROM python:3.9-slim

# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY test_model.py .
# COPY *.pkl . 
# # Copy any model files if you have them locally

# CMD ["python", "-m", "pytest", "test_model.py", "-v"]