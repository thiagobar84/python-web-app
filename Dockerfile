FROM python:3.10-slim

# Instala as dependências do sistema Linux necessárias para geoprocessamento e OpenCV
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    proj-data \
    proj-bin \
    libgeos-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copia os arquivos do projeto para o servidor
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Expõe a porta padrão que a Hugging Face usa para exibir o app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
