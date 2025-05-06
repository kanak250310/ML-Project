# Use official Python image with slim build
FROM python:3.9-slim as builder

# Install essential build tools and clean up afterwards
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    wget \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with CPU-only PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 10 --default-timeout=100 \
    -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Download model files with checksum verification
RUN pip install --no-cache-dir gdown && \
    gdown --id 1GFir7sAkaxLXLeCsPpBE_UBb8wfJlnyX -O AG_SafeStudent.pt && \
    gdown --id 1XGpebvsOQOxuZLZR3Vf4giZ-rX_8dRSN -O ag_news_vocab.pkl && \
    sha256sum AG_SafeStudent.pt > model.sha256 && \
    sha256sum ag_news_vocab.pkl > vocab.sha256

# Final lightweight stage
FROM python:3.9-slim

# Copy only necessary files from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder AG_SafeStudent.pt .
COPY --from=builder ag_news_vocab.pkl .
COPY --from=builder model.sha256 .
COPY --from=builder vocab.sha256 .

# Copy application code
COPY streamlit_app.py .

# Verify model files integrity
RUN sha256sum -c model.sha256 && sha256sum -c vocab.sha256

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose and run
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]