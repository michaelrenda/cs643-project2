FROM ubuntu:latest

MAINTAINER Michael Renda <mr787@njit.edu>


# Install OpenJDK 8 Runtime Environment
RUN apt-get update
RUN apt-get install default-jdk -y
RUN update-alternatives --config java
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6


RUN apt-get update
RUN apt-get -y install python-is-python3
RUN apt-get -y install python3-pip
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN pip install pyspark
RUN pip install pandas

ADD renda_model renda_model
ADD renda_test_model.py renda_test_model.py
ADD ValidationDataset.csv ValidationDataset.csv

ENTRYPOINT python renda_test_model.py