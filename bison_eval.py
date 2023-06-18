#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Attribution-NonCommercial 4.0 International
#https://github.com/facebookresearch/binary-image-selection/blob/main/LICENSE
#

import os
import json


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
        for bison_id in self.params['bison_ids']:
            if self.pred[bison_id] is None:
                continue
            accuracy.append(self.anno[bison_id]['true_image_id'] ==
                            self.pred[bison_id])
        mean_accuracy = np.mean(accuracy)
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
    def __init__(self, pred_filepath=None,pred_results=None):
        if pred_filepath:
            assert os.path.exists(pred_filepath), 'Prediction file does not exist'
            with open(pred_filepath) as fd:
                pred_results = json.load(fd)

        self._data = {result['bison_id']: result['predicted_img_id']
                      for result in pred_results}

    def getBisonIds(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]







