#!/bin/bash

# Use this file in your Dockerfile to run the services

python /app_home/src/modelling/main.py
prefect server start --host 0.0.0.0 --port 4201 &
uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
