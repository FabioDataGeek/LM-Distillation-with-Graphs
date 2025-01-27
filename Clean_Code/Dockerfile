FROM ubuntu:22.04

RUN apt-get update -y
RUN apt-get install pip -y
RUN pip install --user -U nltk
RUN pip install pandas
RUN pip install numpy
RUN pip install tqdm
RUN pip install networkx[default]
RUN pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
RUN pip install transformers
RUN pip install stanza
RUN pip install wn
RUN pip install svgling
RUN pip install -U git+https://github.com/yzhangcs/parser
