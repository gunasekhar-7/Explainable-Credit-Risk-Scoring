FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY src ./src
COPY scripts ./scripts
COPY README.md ./

# Copy pre-trained model artifacts (train separately before building the image)
COPY data/processed ./data/processed

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
