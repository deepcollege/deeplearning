FROM python:3.5.1

RUN mkdir app

WORKDIR app

ADD requirements.txt requirements.txt

RUN pip install -r requirements.txt

CMD ["python3", "DCGAN_advanced.py"]
