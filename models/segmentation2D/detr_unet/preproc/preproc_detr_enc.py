import torch
from torch.utils import model_zoo

checkpoint = model_zoo.load_url("https://dl.fbaipublicfiles.com/detr/detr-r50-panoptic-00ce5173.pth")
detr_dict = checkpoint["model"]

## create a model to check we can load correctly
layer = torch.nn.TransformerEncoderLayer(nhead=8, d_model=256)
trans = torch.nn.TransformerEncoder(num_layers=6, encoder_layer=layer)
trans_dict = trans.state_dict()

state_dict = {}
for detr_key, detr_val in detr_dict.items():
    for trans_key, trans_val in trans_dict.items():
        if trans_key in detr_key and "encoder" in detr_key:
            state_dict[trans_key] = detr_val

trans.load_state_dict(state_dict)

torch.save(state_dict, "trans_init.pth")
