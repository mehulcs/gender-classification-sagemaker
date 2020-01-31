FROM python:3.7.6

RUN apt-get update
RUN apt-get install -y wget nginx

COPY /gender_classification/requirements.txt /opt/ml/code/requirements.txt
WORKDIR /opt/ml/code
RUN pip install --default-timeout=1000 -r requirements.txt

COPY /gender_classification /opt/ml/code

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"
