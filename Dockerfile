FROM tensorflow/tensorflow

RUN apt-get update -y
RUN apt-get install python-tk -y
RUN apt-get clean