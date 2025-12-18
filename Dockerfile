FROM python:3.11.5-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Menjalankan script training saat container dijalankan
CMD ["python", "modelling_tuning.py"]