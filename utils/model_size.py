from sentence_transformers import SentenceTransformer
from torch.quantization import quantize_dynamic
import torch.nn as nn
import torch
from pathlib import Path


image_model = SentenceTransformer('clip-ViT-B-32')
# uncomment to quantize the model
#image_model = quantize_dynamic(image_model, {nn.Linear}, dtype=torch.qint8)

text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')


# uncomment to quantize the model
#text_model = quantize_dynamic(text_model, {nn.Linear}, dtype=torch.qint8)

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

print(f'Image model size: {size_img:.2f} MB \n')
print(f'Text model size: {size_text:.2f} MB')


"""
Size of the models:
Image model size (Non-quantized): 577.2260179519653 MB --> 577.23 MB
Text model size (Non-quantized): 515.5073661804199 MB --> 515.51 MB


"""