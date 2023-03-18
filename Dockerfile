# Stage 1: Builder/Compiler
FROM pytorch/pytorch:latest
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc
COPY requirements.txt /requirements.txt
COPY ./eeggan /eeggan
RUN pip install --no-cache-dir --user -r /requirements.txt
WORKDIR /eeggan/train
CMD ["/bin/bash"]
ENTRYPOINT ["python", "run.py"]
