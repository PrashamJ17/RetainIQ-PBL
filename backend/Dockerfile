FROM python:3.13-slim

WORKDIR /app

# Install dependencies required by XGBoost and SQLite
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install only the backend microservice dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files since backend imports core ML scripts from the root directory
COPY . .

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
