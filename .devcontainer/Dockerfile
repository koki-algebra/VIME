FROM jupyter/datascience-notebook AS cpu

ARG USERNAME=jovyan
COPY library-scripts/common-debian.sh /tmp/library-scripts/
USER root
RUN apt-get update \
		&& groupadd ${USERNAME} \
		&& usermod -g ${USERNAME} -a -G users ${USERNAME} \
		&& bash /tmp/library-scripts/common-debian.sh \
		&& apt-get clean -y && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
# COPY requirements.txt /tmp/pip-tmp/
# RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
#    && rm -rf /tmp/pip-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

USER ${USERNAME}

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu



FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime AS gpu

# default shell
ENV SHELL /bin/bash
SHELL [ "/bin/bash", "-l", "-c" ]

# Create the user
ARG USERNAME=jovyan
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# install basic packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        curl \
    # clear
    && apt-get -y clean \
    && rm -rf /var/lib/apt/lists/*

# install python packages
RUN conda install -c conda-forge ipywidgets jupyterlab \
    && conda install pandas scikit-learn

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME