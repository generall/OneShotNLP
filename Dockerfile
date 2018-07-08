FROM ufoym/deepo:pytorch-py36

WORKDIR /

COPY . /

RUN bash -x install.sh
