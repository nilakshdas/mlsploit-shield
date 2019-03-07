FROM python:2.7

RUN mkdir -p /mnt/input
RUN mkdir -p /mnt/output
RUN mkdir -p /app

ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN bash tools/get_model_checkpoints.sh scratch/checkpoints

CMD ["bash", "run"]
