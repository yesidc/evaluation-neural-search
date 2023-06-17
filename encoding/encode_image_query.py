from sentence_transformers import SentenceTransformer
from PIL import Image
from torch.quantization import quantize_dynamic
import torch.nn as nn
import torch

#Load CLIP model
# 'clip-ViT-B-32' --> 605 MB
# clip-ViT-L-14' --> 1.71 GB
# todo this model should not always be loaded (load asynchronously), but only when the user uploads media.
# todo to consider: barch inference instead of encoding an image at a time.
model = SentenceTransformer('clip-ViT-B-32')
# model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
# text_model = quantize_dynamic(text_model, {nn.Linear}, dtype=torch.qint8)


def process_image(file):
    """
    Encode an image:
    :param file: path to image file
    :return: vector embedding of image
    """
    img = model.encode(Image.open(file))
    img_emb = model.encode(img)
    return img_emb.tolist()


def encode(sentences):
    #Encode text descriptions
    text_emb = text_model.encode(sentences)
    return text_emb
