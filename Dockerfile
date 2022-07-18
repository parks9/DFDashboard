FROM python:3.7-buster

RUN apt-get update && apt-get install -y wget sextractor

RUN pip install astropy pandas scipy matplotlib numpy pyyaml tqdm pyvo fitsio photutils reproject

# install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

# astrometry.net prerequisites
RUN apt-get install -y build-essential curl git file pkg-config swig \
    libcairo2-dev libnetpbm10-dev netpbm libpng-dev libjpeg-dev \
    zlib1g-dev libbz2-dev libcfitsio-dev wcslib-dev

# install astrometry.net
RUN wget http://astrometry.net/downloads/astrometry.net-0.85.tar.gz
RUN tar zxf astrometry.net-0.85.tar.gz
WORKDIR /astrometry.net-0.85
RUN make && make py && make extra && make install

# grab index files index-4207-*
RUN wget -r -nd -np -P /usr/local/astrometry/data/ -A 'index-4207-*.fits' data.astrometry.net/4200/

ENV PATH="$PATH:/usr/local/astrometry/bin"
ENV PYTHONPATH="${PYTHONPATH}:/usr/local/astrometry/lib/python"

WORKDIR /app

# install DFReduce
COPY . .
RUN pip install -e .

RUN python -c "import dfreduce"
