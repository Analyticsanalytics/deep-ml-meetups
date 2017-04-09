apt-get update && apt-get install --no-install-recommends  -y \
    git cmake build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboos
t-iostreams-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev 
liblmdb-dev libutfcpp-dev wget unzip supervisor \
    python \
    python-dev \
    python2.7-dev \
    python3-dev \
    python-virtualenv \
    python-wheel \
	python-tk \
    pkg-config \
    libopenblas-base \
    python-numpy \
    python-scipy \
    python-yaml \
    python-pydot \
    python-nose \
	python-h5py \
	python-skimage \
	python-matplotlib \
	python-pandas \
	python-sklearn \
	python-sympy \
	python-joblib \
        build-essential \
        software-properties-common \
        g++ \
        git \
        wget \
        tar \
        git \
        imagemagick \
        curl \
		bc \
		htop\
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		unzip \
		vim \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
		doxygen \
		less \
        htop \
        procps \
        vim-tiny \
        libboost-dev \
        libgraphviz-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-
ubuntu)
update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

echo "deb http://ppa.launchpad.net/keithw/glfw3/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/fillwave_ext.list
echo "deb-src http://ppa.launchpad.net/keithw/glfw3/ubuntu trusty main" | sudo tee -a /etc/apt/sources.list.d/fillwave_ext.list

sudo add-apt-repository -y ppa:zoogie/sdl2-snapshots
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo add-apt-repository -y ppa:andykimpe/cmake
sudo add-apt-repository -y ppa:h-rayflood/llvm-upper
sudo apt-get -qq -y update


apt-get install -qqy --force-yes libglfw3 libglfw3-dev
apt-get -qyy install build-essential scons pkg-config libx11-dev libxcursor-dev libxinerama-dev libgl1-mesa-dev libglu-dev libas
ound2-dev libpulse-dev libfreetype6-dev libssl-dev libudev-dev libxrandr-dev
apt-get -qqy install mesa-common-dev freeglut3-dev libglfw-dev libglm-dev libglew1.6-dev xorg-dev libglu1-mesa-dev libsdl2-dev
apt-get -qq -y install libsdl2-dev libsdl2-ttf-dev libalut-dev libpng12-dev

cd /tmp/
wget https://cmake.org/files/v3.8/cmake-3.8.0-rc4.tar.gz
tar -xvf cmake-3.8.0-rc4.tar.gz
cd /tmp/cmake-3.8.0-rc4
./bootstrap
make
make install
