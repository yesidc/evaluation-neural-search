# Evaluation-neural-search

This repository contains the code for the evaluation of text-to-image retrieval system. The evaluation is based on the COCO-BISON dataset. The dataset is available at <a href="http://www.hexianghu.com/bison/">COCO-BISON dataset</a>. The dataset contains a total of 54253 examples, 38680 unique images, and 45218 unique captions.
To evaluate the information retrieval system, we compute the BISON score to determine the ability of the system to match linguistic content with fine-grained visual structure (see Hu at al., 2019). 

## Implementation
The implementation is based on the <a href="https://github.com/facebookresearch/binary-image-selection/blob/main">BISON repository</a> by Hu et al. (2019). The repository contains the code for the evaluation of the BISON score which was adapted to the needs of this project. 

## Requirements

- torch  2.0.0 
- python 3.9.7 
- numpy 1.22.3
- matplotlib 3.5.1 
- pandas 1.5.3
- seaborn 0.12.2
- scipy 1.11.1
- pillow 9.0.1 
- sentence-transformers 2.2.2
- json5 0.9.10 
- tqdm 4.62.3

## Usage
Run the following command to evaluate the BISON score of a text-to-image retrieval system. 
```
python main.py --anno_bison_path <path_to_bison_annotations> --create_pred_file <True/False> --pred_path <path_to_prediction_file> --val_captions_path <path_to_validation_captions_file> --val_img_path <path_to_validation_images_directory>
```

Arguments:
- `--anno_bison_path`: Path to the bison annotation file. Default: `./annotations/bison_annotations.cocoval2014.json`
- `--create_pred_file`: Create a prediction file prediction.json. Default: `False`.
- `--pred_path`: Path to the prediction file. Default: `./predictions/prediction.json`
- `--val_captions_path`: Path to the validation captions file.
- `--val_img_path`: Path to the directory that contains the validation images.

To be able to generate a prediction file you must provide a both a path to the validation captions file and the validation images directory. The validation captions file and images (COCO-2014) are available at <a href="https://cocodataset.org/#download">COCO dataset</a>. 




## References
- H. Hu, I. Misra, and L. Van Der Maaten. Evaluating text-to-image matching using binary
image selection (bison). In Proceedings of the IEEE/CVF International Conference on
Computer Vision Workshops, pages 0–0, 2019.
- <a href="http://www.hexianghu.com/bison/">COCO-BISON dataset</a>

 