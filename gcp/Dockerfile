# SNN training image for Vertex AI custom training (GPU). cu128 covers Ada (L4),
# Ampere (A100) and Blackwell (sm_120). The conductance Diehl & Cook 6400/60k run that
# is impractical on CPU finishes here.
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
 && pip3 install --no-cache-dir snntorch bindsnet numpy

# default = the literature 95% config; override via Vertex env or --args
ENV NORD_M=6400 NORD_TRAIN=60000 NORD_TEST=10000 PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "eth_mnist_bindsnet.py", "--gpu"]
