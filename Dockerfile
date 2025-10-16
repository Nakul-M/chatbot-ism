# 🧱 Base image with PyTorch (CPU)
FROM python:3.11-slim
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu



# 🏗️ Set working directory
WORKDIR /app

# 🐍 Ensure Python packages install smoothly
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 📦 Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# 🧾 Copy application files
COPY . .

# 🔥 Expose FastAPI port
EXPOSE 8000

# 🏃‍♂️ Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
