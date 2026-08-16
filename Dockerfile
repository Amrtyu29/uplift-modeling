# Serving image: API + dashboard. Training happens outside the container; the
# trained bundle is copied in, so the image has no reason to carry the datasets.
FROM python:3.11-slim

# LightGBM needs libgomp at runtime — the wheel links against it but does not
# vendor it, and its absence shows up as an import error rather than a build one.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so an application-code change does not invalidate the
# dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY models/ ./models/

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    UPLIFT_DATASET=hillstrom \
    PORT=8000

# Fails the container if the model artifact is missing or unloadable, rather
# than letting it serve 503s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

EXPOSE 8000
# Honour $PORT so the image runs unchanged on Render / Fly.io / Cloud Run.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
