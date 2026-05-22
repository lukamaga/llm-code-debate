FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p visualizations results

EXPOSE 5050

ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

CMD ["python", "-m", "src.web.app"]
