import torch

import numpy as np

from utils.forward import w2v


def predict(text: str, model):
    model.eval()
    text_vector = torch.from_numpy(w2v(text).astype(np.float32))
    output = model(text_vector)
    _, pred_y = torch.max(output, axis=-1)
    pred_y = pred_y.item()
    proba = output[0][pred_y].item()
    if pred_y == 0:
        return "Negativo", proba
    else:
        return "Positivo", proba
