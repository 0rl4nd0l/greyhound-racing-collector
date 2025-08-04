# Multi-stage build for production deployment
FROM python:3.11-slim-bullseye as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    unzip \
    xvfb \
    # Chrome dependencies
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 \
    libatk1.0-0 \
    libc6 \
    libcairo2 \
    libcups2 \
    libdbus-1-3 \
    libexpat1 \
    libfontconfig1 \
    libgbm1 \
    libgcc1 \
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libstdc++6 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrandr2 \
    libxrender1 \
    libxss1 \
    libxtst6 \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-install and cache ChromeDriver using webdriver-manager
RUN python -m webdriver_manager.chrome \
    && echo "ChromeDriver installed and cached successfully"

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

# Set default environment for Chrome in headless mode
ENV DISPLAY=:99
ENV CHROME_BIN=/usr/bin/google-chrome-stable
ENV CHROMEDRIVER_PATH=/home/appuser/.wdm/drivers/chromedriver

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from drivers import setup_selenium_driver_path; setup_selenium_driver_path(); print('ChromeDriver OK')" || exit 1

# Expose port for Flask app
# Use environment variable for the port
EXPOSE 5002

# Default command
CMD ["sh", "-c", "python -m flask run --host=0.0.0.0 --port=${PORT:-5002}"]

# Development stage
FROM base as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install testing dependencies
RUN pip install --no-cache-dir -r requirements-test.txt

USER appuser

# Production stage
FROM base as production

# Copy only necessary files for production
COPY --chown=appuser:appuser . .

# Run smoke test to verify ChromeDriver setup
RUN python tests/test_chromedriver_smoke.py

CMD ["sh", "-c", "python -m flask run --host=0.0.0.0 --port=${PORT:-5002}"]
