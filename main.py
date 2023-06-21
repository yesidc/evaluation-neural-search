import argparse
from bison_eval import BisonEval, Annotation, Prediction, ValidationCaptions
from predictions.geneate_predictions import compute_predictions
from logger_evaluation import logger
import logging

"""
val_img_path = '/Users/yesidcano/Downloads/val2014/'
val_captions_path = /Users/yesidcano/Downloads/annotations/captions_val2014.json
anno_bison_path = /Users/yesidcano/repos/binary-image-selection/annotations/bison_annotations.cocoval2014.json
"""
# create logger
logger = logging.getLogger('evaluation')


def _command_line_parser():
    parser = argparse.ArgumentParser()
    default_anno = './annotations/bison_annotations.cocoval2014.json'
    default_pred = './predictions/prediction.json'
    parser.add_argument('--anno_bison_path', default=default_anno,
                        help='Path to the bison annotation file')

    parser.add_argument('--create_pred_file', default=False,
                        help='Create a prediction file prediction.json')
    parser.add_argument('--pred_path', default=default_pred,
                        help='Path to the prediction file')

    parser.add_argument('--val_captions_path',
                        help='Path to the file that contains the captions')
    parser.add_argument('--val_img_path',
                        help='Path to the directory that contains the validation images')

    return parser


def main(args):
    anno_bison = Annotation(args.anno_bison_path)
    validation_captions = ValidationCaptions(args.val_captions_path)
    if args.create_pred_file:
        pred_results = compute_predictions(anno_bison,
                                           args.val_img_path,
                                           validation_captions)
        pred = Prediction(pred_results=pred_results)
    else:
        pred = Prediction(pred_filepath=args.pred_path)
    bison = BisonEval(anno_bison, pred)
    mean_accuracy = bison.evaluate()
    print(mean_accuracy)
    #logger.info(f'Mean BISON accuracy on {anno_bison.dataset}: {mean_accuracy}%')



if __name__ == '__main__':
    parser = _command_line_parser()
    args = parser.parse_args()
    main(args)
