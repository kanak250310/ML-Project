FROM python:3.10-alpine

# Set environment variables to prevent interactive prompts during installs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies needed for pip and Python packages
RUN apk add --no-cache \
    build-base \
    libffi-dev \
    jpeg-dev \
    zlib-dev \
    freetype-dev \
    lcms2-dev \
    openjpeg-dev \
    tiff-dev \
    tk-dev \
    tcl-dev \
    python3-dev \
    musl-dev \
    bash \
    curl \
    git

# Upgrade pip
RUN pip install --upgrade pip

# Copy requirements file first and install dependencies (numpy first)
COPY requirements.txt .

RUN pip install numpy && \
    pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

