from sentence_transformers import SentenceTransformer
from PIL import Image
from torch.quantization import quantize_dynamic
import torch.nn as nn
import torch
from logger_evaluation import logger
import logging


# create logger
logger = logging.getLogger('evaluation')

#Load CLIP model
# 'clip-ViT-B-32' --> 605 MB


model = SentenceTransformer('clip-ViT-B-32')
model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')
text_model = quantize_dynamic(text_model, {nn.Linear}, dtype=torch.qint8)
logger.info('CLIP model has been quantized.')

def encode_image(file):
    """
    Encode an image:
    :param file: path to image file
    :return: vector embedding of image
    """
    img_emb = model.encode(Image.open(file))

    return img_emb


def encode_query(query):
    """
    Encode a text query:
    :param query: string
    :return: text embedding
    """
    #Encode text descriptions
    text_emb = text_model.encode(query)
    return text_emb
