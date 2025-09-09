FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies for compiling packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY assistant.py .
COPY app.py .
COPY data ./data
COPY visualization_img ./visualization_img

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8080", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false"]
