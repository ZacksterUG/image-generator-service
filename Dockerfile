FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as conda

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Установка Miniconda
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Принятие условий использования каналов
RUN conda config --set always_yes yes --set changeps1 no && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Создание окружения с Python 3.12
RUN conda create -n base_env python=3.12 -y

# Финальный образ
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Копирование Conda из предыдущего stage
COPY --from=conda /opt/conda /opt/conda

# Настройка окружения
ENV PATH=/opt/conda/envs/base_env/bin:$PATH
ENV CONDA_DEFAULT_ENV=base_env
ENV CONDA_PREFIX=/opt/conda/envs/base_env

# Установка системных зависимостей если нужны
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . model/

ENV PYTHONPATH=/app

CMD ["python", "model/main.py"]