DATA_VOLUME="-v $(pwd)/..:/data"
HOME_VOLUME="-v $HOME:$HOME"
docker run --rm --runtime=nvidia -it --net=host --ipc=host ${DATA_VOLUME} ${HOME_VOLUME} --name=satnet satnet
