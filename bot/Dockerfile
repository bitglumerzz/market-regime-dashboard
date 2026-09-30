FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY regime_bot ./regime_bot
RUN useradd -m bot && mkdir -p /app/data && chown bot /app/data
USER bot
EXPOSE 8081
CMD ["python", "-m", "regime_bot"]
