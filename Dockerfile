FROM nvidia/cuda:11.3.1-base-ubuntu18.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y wget git 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /root/miniconda3
RUN rm -f miniconda.sh
ENV PATH=/root/miniconda3/bin:$PATH
RUN conda update pip

RUN git clone -b docker https://github.com/andrey-okhotin/star-shaped.git

RUN pip install py7zr gdown
RUN rm -rf star-shaped/datasets
RUN gdown --fuzzy https://drive.google.com/file/d/1ndXOmbNXR6pwoJ5qs1gVP0eAKU_RAl6E/view?usp=sharing
RUN py7zr x datasets.7z && rm datasets.7z && mv datasets star-shaped/datasets

RUN conda create -n base_env python=3.9
SHELL ["conda", "run", "-n", "base_env", "/bin/bash", "-c"]
RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -r star-shaped/requirements.txt
RUN git clone https://github.com/gregversteeg/NPEET.git && cd NPEET && pip install . && cd ../ && rm -rf NPEET

# Set the working directory
WORKDIR /star-shaped

# Set the entrypoint
# ENTRYPOINT [ "bash" ]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base_env", "python", "lib/run_pipeline.py" ]