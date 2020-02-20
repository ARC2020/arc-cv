FROM tensorflow/tensorflow:1.0.0-rc1-py3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./DATA
COPY Dockerfile ./Dockerfile
COPY encoder ./encoder
COPY hypes ./hypes
COPY submodules ./submodules
COPY decoder ./decoder
COPY docu ./docu
COPY evals ./evals
COPY incl ./incl
COPY licenses ./licenses
COPY requirements.txt ./requirements.txt
COPY main.py ./main.py
COPY inputs ./inputs
COPY optimizer ./optimizer
COPY RUNS ./RUNS

CMD [ "python", "demo.py", "--input_image", "DATA/demo/demo.png" ]