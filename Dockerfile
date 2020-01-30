FROM python:3.7.6

RUN apt-get update
RUN apt-get install -y wget nginx

COPY /container /opt/ml/code

WORKDIR /opt/ml/code

RUN pip install -r requirements.txt
