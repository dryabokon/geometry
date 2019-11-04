FROM python:3.7

RUN apt-get update

RUN apt-get -y install libgtk2.0-dev pkg-config

RUN apt-get -y install build-essential cmake libjpeg-dev libpng-dev libavformat-dev

RUN pip install scipy dlib scikit-learn matplotlib Pillow scikit-image opencv-python numpy

