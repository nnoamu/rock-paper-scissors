FROM python:3.10

WORKDIR /app

# Másold be a requirements fájlt
COPY requirements.txt .

# Telepítsd a függőségeket
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Másold be az összes fájlt
COPY . .

# Gradio port
EXPOSE 7860

# Indítsd az appot
CMD ["python", "src/app.py"]