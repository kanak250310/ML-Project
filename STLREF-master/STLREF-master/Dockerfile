# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# FIRST install regular requirements
RUN pip install --no-cache-dir -r requirements.txt

# THEN install CPU-only PyTorch with EXPLICIT URL
RUN pip install --no-cache-dir \
    https://download.pytorch.org/whl/cpu/torch-2.0.1%2Bcpu-cp39-cp39-linux_x86_64.whl \
    https://download.pytorch.org/whl/cpu/torchvision-0.15.2%2Bcpu-cp39-cp39-linux_x86_64.whl

# Make port 8501 available
EXPOSE 8501

# Run streamlit_app.py when the container launches
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]