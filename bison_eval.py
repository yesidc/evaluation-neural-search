#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Attribution-NonCommercial 4.0 International
# https://github.com/facebookresearch/binary-image-selection/blob/main/LICENSE

# NOTE: This code was altered to serve the purpose of the project. The original code can be found at the link above.
# ValidationCaptions class was added to read the validation captions file. Prediction class was altered to accept a
# dictionary as input. Annotation class was altered. Added __iter__ the functions main() nad _commandline_parser()
# were moved to main.py, and modified to serve the purpose of the project.


import os
import json
from tqdm import tqdm
import numpy as np


class BisonEval:
    def __init__(self, anno, pred):
        if pred.getBisonIds() != anno.getBisonIds():
            print('[Warning] The prediction does not' +
                  'cover the entire set of bison data.' +
                  'The evaluation is running on the {}'.format(
                      len(pred.getBisonIds())) +
                  'subset from prediction file.')
        self.params = {'bison_ids': pred.getBisonIds()}
        self.anno = anno
        self.pred = pred

    def evaluate(self):
        accuracy = []
        wrong_predictions = []
        correct_predictions = []
        for bison_id in tqdm(self.params['bison_ids']):
            current_datapoint = None
            if self.pred[bison_id] is None:
                continue
            result = self.anno[bison_id]['true_image_id'] == self.pred[bison_id]['predicted_img_id']
            accuracy.append(result)

            current_datapoint = {
                    'bison_id': bison_id,
                    'high_similarity': self.pred[bison_id]['high_similarity'],
                    'low_similarity': self.pred[bison_id]['low_similarity'],
                    'true_image_id': self.anno[bison_id]['true_image_id'],
                    'predicted_img_id': self.pred[bison_id]['predicted_img_id'],
            }
            # if prediction is wrong
            if not result:
                wrong_predictions.append(current_datapoint)
            else:
                correct_predictions.append(current_datapoint )

        # save correct predictions to json file
        with open('./predictions/correct_predictions.json', 'w') as fd:
            json.dump(correct_predictions, fd)
        # compute BISON accuracy
        mean_accuracy = np.mean(accuracy)
        # save wrong predictions to json file
        with open('./predictions/wrong_predictions.json', 'w') as fd:
            json.dump(wrong_predictions, fd)

        print("[Result] Mean BISON accuracy on {}: {:.2f}%".format(
            self.anno.dataset, mean_accuracy * 100)
        )
        return mean_accuracy


class ValidationCaptions:
    def __init__(self, val_captions_path):
        assert os.path.exists(val_captions_path), 'Validation file does not exist'
        with open(val_captions_path) as fd:
            validation_captions = json.load(fd)
        # id refers to the caption id
        self._data = {anno['id']: anno['caption'] for anno in validation_captions['annotations']}

    def __getitem__(self, key):
        return self._data[key]


class Annotation:
    def __init__(self, anno_filepath):
        assert os.path.exists(anno_filepath), 'Annotation file does not exist'
        with open(anno_filepath) as fd:
            anno_results = json.load(fd)
        self._data = {res['bison_id']: res for res in anno_results['data']}
        self.dataset = "{}.{}".format(anno_results['info']['source'],
                                      anno_results['info']['split'])

    def getBisonIds(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)


class Prediction:
    def __init__(self, pred_filepath=None, pred_results=None):
        if pred_filepath:
            assert os.path.exists(pred_filepath), 'Prediction file does not exist'
            with open(pred_filepath) as fd:
                pred_results = json.load(fd)

        self._data = {result['bison_id']: result
                      for result in pred_results}

    def getBisonIds(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]
