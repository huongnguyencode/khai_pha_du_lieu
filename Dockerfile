FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . /workspace
RUN PYTHONPATH=src python3 scripts/01_build_nhanes_database.py \
    && python3 -m unittest discover -s tests -v

CMD ["bash", "-lc", "PYTHONPATH=src python3 scripts/01_build_nhanes_database.py && PYTHONPATH=src python3 scripts/02_train_w1.py"]

