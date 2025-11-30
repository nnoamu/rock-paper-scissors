FROM python:3.10

WORKDIR /app

COPY requirements.txt .

# Verbose install hogy lássuk a dependency resolutiont
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -v gradio==6.0.1 && \
    pip list | grep -i gradio && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "=== FINAL GRADIO VERSION ===" && \
    pip list | grep -i gradio

COPY . .

EXPOSE 7860

CMD ["python", "src/app.py"]