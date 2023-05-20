import PIL.Image as Image
import numpy as np

def arr2img(arr: np.array) -> Image:
    """
    Params:
        arr: (C, H, W)
    """
    # if channel is 1, save as gray scale
    if arr.shape[0] == 1:
        arr = arr.squeeze()
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(arr.transpose(1, 2, 0).astype(np.uint8))
    return img
