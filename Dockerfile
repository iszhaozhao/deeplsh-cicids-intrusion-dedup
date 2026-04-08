FROM --platform=$TARGETPLATFORM python:3.9-bullseye

# System deps for scientific Python stacks and packages that may compile (scipy/fasttext/spacy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install Python deps first for better layer caching.
COPY code/requirements.txt /app/code/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /app/code/requirements.txt \
    # Required by code/python-packages/lsh_search.py but not listed in requirements.txt
    && python -m pip install fnvhash

# Copy the repo last to keep dependency layer cacheable.
COPY . /app

CMD ["python", "code/run.py", "list"]

