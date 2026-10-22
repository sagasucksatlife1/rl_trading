FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt yfinance requests
COPY . .
RUN python -m pytest -q
# Daily job: mount runs/ (models) and live_state/ (decision log, paper account) as volumes.
# Secrets (ALPACA_KEY_ID / ALPACA_SECRET_KEY) come from the environment, never the image.
ENTRYPOINT ["python", "scripts/shadow_run.py"]
CMD ["--help"]
