import streamlit as st
import torch, torchvision
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import numpy as np

from settings import Dir, Params
from cnn_training.plot_results import plot_closest

st.title('Image retrieval')

query = st.text_input('Query', 'hair')
st.write('The current movie title is', query)

state_dict = torch.load(f"cnn_training/models/resnet18_ny.pt")
resnet = torchvision.models.resnet18()
resnet.fc = torch.nn.Linear(resnet.fc.in_features, Params.dim_embedding) # the model should output in the word vector space
resnet = resnet.to("cuda")
resnet.load_state_dict(state_dict["model_state_dict"])

import io
fig = plot_closest(query)
io_buf = io.BytesIO()
fig.savefig(io_buf, format='raw')
io_buf.seek(0)
img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
io_buf.close()

st.image(img_arr)
