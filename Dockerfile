
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONPATH=/app
ENV PORT=5001

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m models.train_model

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]
