# Recipe for running the demo in a Docker container on any server or cloud:
#   docker build -f deploy/Dockerfile -t drug-review-sentiment .
#   docker run -p 8501:8501 drug-review-sentiment

# 1. Start from a small Linux image with Python 3.12 (the version the models were built with)
FROM python:3.12-slim

# 2. Run the app as a normal user, not as admin (safer)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

# 3. Install the exact package versions used in training (CPU-only PyTorch, no GPU here)
COPY --chown=user deploy/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# 4. Copy in the app code (the models are downloaded from Hugging Face Hub at first use)
COPY --chown=user src/ src/
COPY --chown=user app/ app/
COPY --chown=user .streamlit/ .streamlit/
RUN mkdir -p models

# 5. Start the web page on port 8501
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
