FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/Backend/src

WORKDIR /app

EXPOSE 7860

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY Backend ./Backend
COPY Frontend ./Frontend
COPY assets ./assets
COPY README.md .
COPY LICENSE .

CMD ["streamlit", "run", "Frontend/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]