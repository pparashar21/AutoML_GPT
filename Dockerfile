

# ---------- base image ----------
FROM python:3.10-slim

# ---------- system setup ----------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    PIP_NO_CACHE_DIR=1

# ---------- OpenAI credentials (injected via --build-arg or docker run -e) ----------
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

# Optional: avoid warnings about OpenBLAS threading in containers
ENV OMP_NUM_THREADS=1

WORKDIR /app

# ---------- install Python deps first (leverages Docker layer caching) ----------
COPY requirements.txt /app/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------- copy project source ----------
COPY . /app

# Streamlit listens on 8501 by default
EXPOSE 8501

# ---------- entrypoint ----------
CMD ["streamlit", "run", "src/components/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]