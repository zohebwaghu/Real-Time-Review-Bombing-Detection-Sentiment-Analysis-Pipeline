FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY producer.py .

CMD ["python", "producer.py", \
     "--file", "/data/review.json", \
     "--topic", "yelp-reviews", \
     "--rate", "100", \
     "--bootstrap", "kafka:29092"]
