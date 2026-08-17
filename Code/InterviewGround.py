FROM python:3.13-slim

# Set environment settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL="http://host.docker.internal:11434"

# Set working directory inside container
WORKDIR /app

# Install basic compiler tools required by ChromaDB
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script
COPY InterviewGround.py .

# Open Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "InterviewGround.py", "--server.port=8501", "--server.address=0.0.0.0"]