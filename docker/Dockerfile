FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
RUN pip install setproctitle

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME
ARG HOME_DIR

RUN addgroup --gid ${GROUP_ID} ${USER_NAME} || groupmod -n ${USER_NAME} $(getent group ${GROUP_ID})
RUN apt-get -q update; apt-get -q -y install sudo vim
RUN conda install -y -q jupyter matplotlib
RUN adduser --quiet --disabled-password --system --no-create-home --uid ${USER_ID} --gid ${GROUP_ID} --gecos '' --shell /bin/bash ${USER_NAME}
RUN usermod -d ${HOME_DIR} ${USER_NAME}
RUN adduser --quiet ${USER_NAME} sudo ; echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /data
WORKDIR /data
