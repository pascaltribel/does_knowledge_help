FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG userPort=9119
ARG userName=ptribel
ARG userGID=1016
ARG userID=1016

SHELL ["/bin/bash", "-c"]

# Install system deps
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev git\
    build-essential ca-certificates wget && \
    rm -rf /var/lib/apt/lists/*

# Optionally symlink python and pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && rm get-pip.py

# Install PyTorch + torchvision (CUDA 11.8 build)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install scipy
RUN export PATH=$PATH:/home/ptribel/.local/bin
RUN pip install jupyter
RUN pip install seisbench
RUN pip install neuraloperator
RUN pip install seaborn
#RUN pip install numba
#RUN pip install h5py pandas mpi4py numba-mpi
#RUN pip install devito
RUN pip install tqdm
RUN pip install scikit_learn
#RUN pip install opencv-python
#RUN pip install tsfresh
#RUN pip install pyawd
RUN pip install xgboost
#RUN pip install pywavelets
RUN pip install scikit-image
#RUN pip install tsfel
#RUN pip install "sktime[all-extras]"
#RUN pip install tslearn
RUN pip install deepwave
RUN pip install torchsummary
RUN echo "export SEISBENCH_CACHE_ROOT=\"/home/ptribel/shared_data/seisbench\"" >> .bashrc
RUN git clone https://github.com/HPCSys-Lab/simwave.git
RUN pip3 install -e ./simwave
RUN pip install skorch


RUN groupadd -g $userGID $userName
RUN useradd -u $userID -d /home/$userName -ms /bin/bash -g $userGID -G sudo,$userName -p $(openssl passwd -1 abc123) $userName
RUN usermod -aG sudo $userName

# Add volume to allow data exchange with the host machine
RUN mkdir /home/$userName/shared_data
RUN chown $userName:$userName /home/$userName/shared_data

RUN export PATH=$PATH:/home/ptribel/.local/bin

USER $userName
WORKDIR /home/$userName
VOLUME /home/$userName/shared_data
EXPOSE $userPort
LABEL org.opencontainers.image.authors="pascal.tribel@ulb.be"

WORKDIR /home/ptribel/shared_data/

#CMD ["/bin/sh", "-c", "python3 Pipelines_NO.py"]
