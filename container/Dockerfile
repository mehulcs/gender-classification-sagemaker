FROM python:3.7.6

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget nginx && \
    rm -rf /var/lib/apt/lists/*

COPY /gender_classification/requirements.txt /opt/ml/code/requirements.txt
WORKDIR /opt/ml/code
RUN pip install --default-timeout=1000 -r requirements.txt && \
    rm -rf /root/.cache

COPY /gender_classification /opt/ml/code

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
