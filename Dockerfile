FROM python:3.10-slim

WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY . .

# Exponer el puerto que utiliza Gradio por defecto
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]