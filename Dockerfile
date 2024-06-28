FROM python:3.12.4-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget 

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# replace with training command
CMD bash
