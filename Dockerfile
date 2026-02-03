FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p visualizations results

# Expose port
EXPOSE 5050

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

# Run the web server
CMD ["python", "-m", "src.web.app"]
