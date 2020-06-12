FROM python:3.8-slim-buster

RUN apt update \
    && apt install tesseract-ocr tesseract-ocr-eng -y \
    && apt clean && apt autoremove \
    && rm -rf /var/lib/apt/lists* /tmp/* /var/tmp/*

RUN export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

ADD . /app
WORKDIR /app
RUN pip install -r ./requirements.txt

EXPOSE 5000
CMD [ "python", "app.py" ]
