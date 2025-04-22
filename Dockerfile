FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy package files
COPY . .
RUN pip install --no-cache-dir -e .
