FROM python:3.10.13-slim

WORKDIR /app_home

COPY ./requirements.txt /app_home/requirements.txt
RUN apt-get update && apt-get install -y gcc python3-dev \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r /app_home/requirements.txt

COPY ./src /app_home/src
COPY ./config /app_home/src/modelling/config
RUN mkdir /app_home/data
COPY ./bin/run_services.sh /app_home/run_services.sh

EXPOSE 8001
EXPOSE 4201

RUN chmod 777 /app_home/run_services.sh

CMD ["/app_home/run_services.sh"]