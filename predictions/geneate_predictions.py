from encoding.encode_image_query import encode_image, encode_query
from sentence_transformers import util
import numpy as np
import json
from tqdm import tqdm
from logger_evaluation import logger
import logging


# create logger
logger = logging.getLogger('evaluation')


def compute_similarity(img_1,img_2, query_emb, bison_id):

    predicted_img = None
    #Compute cosine similarity between all pairs
    # similarity score is between -1 and 1
    similarity_1 = util.cos_sim(query_emb,
                              np.array(img_1['vector_embedding'], dtype=np.float32)).item()


    similarity_2 = util.cos_sim(query_emb,
                              np.array(img_2['vector_embedding'], dtype=np.float32)).item()


    if similarity_1 > similarity_2:
        predicted_img = {
            'predicted_img_id': img_1['img_id'],
            'bison_id': bison_id,
            'high_similarity': similarity_1,
            'low_similarity': similarity_2
        }
    elif similarity_2 > similarity_1:
        predicted_img = {
            'predicted_img_id': img_2['img_id'],
            'bison_id': bison_id,
            'high_similarity': similarity_2,
            'low_similarity': similarity_1
        }
    else:
        # if similarity_1 == similarity_2 then the sample is discarded
        predicted_img = {
            'predicted_img_id': '',
            'bison_id': bison_id,
            'high_similarity': similarity_2,
            'low_similarity': similarity_1
        }
        logger.warning (f'Similarity is equal {predicted_img}')
        print(f'Similarity is equal {predicted_img}')

    return predicted_img



def compute_predictions(anno_bison, val_img_path, validation_captions):
    # path to image folder must end with /
    
    prediction = []

    for b_id in tqdm(anno_bison):
        b_data = anno_bison[b_id]
        img_1 = {}
        img_2 = {}
        # encode candidate images
        img_1['vector_embedding'] = encode_image(val_img_path + b_data['image_candidates'][0]['image_filename'])
        img_1['img_id'] = b_data['image_candidates'][0]['image_id']

        img_2['vector_embedding'] = encode_image(val_img_path + b_data['image_candidates'][1]['image_filename'])
        img_2['img_id'] = b_data['image_candidates'][1]['image_id']

        # encode query
        query_emb = encode_query(validation_captions[b_data['caption_id']])
        # compute similarity
        predicted_img = compute_similarity(img_1,img_2, query_emb, b_id)
        prediction.append(predicted_img)

    # save prediction to json file
    with open('./predictions/prediction.json', 'w') as outfile:
        json.dump(prediction, outfile)
    print('Predictions saved to prediction.json')
    return prediction
        



