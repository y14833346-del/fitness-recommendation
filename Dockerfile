FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
