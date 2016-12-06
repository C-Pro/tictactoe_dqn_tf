FROM gcr.io/tensorflow/tensorflow
MAINTAINER Sergey Melekhin <sergey@melekhin.me>

RUN apt-get update
RUN apt-get install -y build-essential gfortran
RUN apt-get install -y python3 python3-pip python3-dev
RUN apt-get install -y libblas-dev liblapack-dev
RUN pip3 install scipy && \
 pip3 install tensorflow && \
 pip3 install tflearn
RUN mkdir /ttt
ADD *.py /ttt/
VOLUME ["/tmp"]
WORKDIR /tmp
CMD ["python3","/ttt/learn.py"]
