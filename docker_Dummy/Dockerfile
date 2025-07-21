#trainnig stage
FROM python:3.11-slim as trainer
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir  -r requirements.txt
COPY train.py .
RUN python train.py

#stage2 serving model

FROM python:3.11-slim 
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py .
COPY --from=trainer /app/model.pkl .
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
