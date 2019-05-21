docker image build \
	--build-arg USER_ID=$(id -u ${USER}) \
	--build-arg GROUP_ID=$(id -g ${USER}) \
	--build-arg USER_NAME=$(whoami) \
	--build-arg HOME_DIR=$HOME \
	-t satnet .
