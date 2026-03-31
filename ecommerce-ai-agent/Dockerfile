FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for Prophet and scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching — only reinstalls on requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Ensure required directories exist inside the container
RUN mkdir -p models/saved data/raw data/processed mlruns

# Streamlit port
EXPOSE 8501

# Start the dashboard on container launch
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
