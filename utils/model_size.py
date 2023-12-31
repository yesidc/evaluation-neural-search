from sentence_transformers import SentenceTransformer
from torch.quantization import quantize_dynamic
import torch.nn as nn
import torch
from pathlib import Path
from logger_evaluation import logger


image_model = SentenceTransformer('clip-ViT-B-32')
# uncomment\comment out to quantize or not the model
image_model = quantize_dynamic(image_model, {nn.Linear}, dtype=torch.qint8)

text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


# uncomment\comment out to quantize or not the model
text_model = quantize_dynamic(text_model, {nn.Linear}, dtype=torch.qint8)

state_dict_img = image_model.state_dict()
state_dict_text = text_model.state_dict()

# save the state dict
tmp_path_img = Path('img_model.pt')
torch.save(state_dict_img, tmp_path_img)

tmp_path_text = Path('text_model.pt')
torch.save(state_dict_text, tmp_path_text)

# Calculate the size in MB

size_img = Path(tmp_path_img).stat().st_size / (1024 * 1024)
size_text = Path(tmp_path_text).stat().st_size / (1024 * 1024)

# delete temporary files
tmp_path_img.unlink()
tmp_path_text.unlink()

# change log message accordingly
logger.info(f'Image model size (Quantized): {size_img:.2f} MB')
logger.info(f'Text model size (Quantized): {size_text:.2f} MB')


"""
Size of the models:
Image model size (Non-quantized):  577.23 MB
Text model size (Non-quantized): 515.51 MB

Image model size (Quantized): 224.46 MB
Text model size (Quantized): 392.91 MB
"""
