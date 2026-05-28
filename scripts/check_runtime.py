from __future__ import annotations
import sys
import streamlit as st
import torch
print('python', sys.version)
print('streamlit', st.__version__)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
