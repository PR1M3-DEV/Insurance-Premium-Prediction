# Motor Insurance Premium Prediction API
#
# Build:
#   docker build -t insurance-premium-api .
# Run:
#   docker run -p 8000:8000 insurance-premium-api
#
# Requires artifacts/model.pkl to exist on the host BEFORE building
# (run model_training/train.py first) — it's excluded from git via
# .gitignore but copied into the image from the local build context.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what's needed to serve predictions — no raw data, no training
# script, no tests in the production image.
COPY app.py .
COPY src/ src/
COPY conf/ conf/
COPY artifacts/ artifacts/

ENV APP_ENV=prod
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]