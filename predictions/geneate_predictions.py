from encoding.encode_image_query import encode_image, encode_query
from sentence_transformers import util
import numpy as np
import json
import pandas as pd

def load_coco_val_caption(path_val_captions):
    # load caption COCO val 2014
    with open(path_val_captions) as json_file:
        val_captions_dic = json.load(json_file)
    df_val_captions_dic = pd.DataFrame(val_captions_dic['annotations'])
    del val_captions_dic
    df_val_captions_dic.set_index('id', inplace=True)
    return df_val_captions_dic

def compute_vector_represetation(anno_filepath, val_img_path):
    # path to image folder must end with /
    
    prediction = []
    # load BISON dataset
    with open(anno_filepath) as fd:
        anno_results = json.load(fd)
    bison_data = {res['bison_id']: res for res in anno_results['data']}

    # retrieve caption from COCO val 2014
    df_val_captions_dic = load_coco_val_caption('/Users/yesidcano/Downloads/annotations/captions_val2014.json')

    for b_id, b_data in bison_data.items():
        img_1 = {}
        img_2 = {}
        # encode candidate images
        img_1['vector_embedding'] = encode_image(val_img_path + b_data['image_candidates'][0]['image_filename'])
        img_1['img_id'] = b_data['image_candidates'][0]['image_id']

        img_2['vector_embedding'] = encode_image(val_img_path + b_data['image_candidates'][1]['image_filename'])
        img_2['img_id'] = b_data['image_candidates'][1]['image_id']

        # encode query
        query_emb = encode_query(df_val_captions_dic.loc[b_data['caption_id']]['caption'])
        # compute similarity
        predicted_img = compute_similarity(img_1,img_2, query_emb, b_id)
        prediction.append(predicted_img)

        if b_id == 3:
            break
    # save prediction to json file
    with open('prediction.json', 'w') as outfile:
        json.dump(prediction, outfile)
    print('Predictions saved to prediction.json')
        




def compute_similarity(img_1,img_2, query_emb, bison_id):

    predicted_img = None
    #Compute cosine similarity between all pairs
    # similarity score is between -1 and 1
    similarity_1 = util.cos_sim(query_emb,
                              np.array(img_1['vector_embedding'], dtype=np.float32)).item()
    # similarity_1 = similarity_1.item()

    similarity_2 = util.cos_sim(query_emb,
                              np.array(img_2['vector_embedding'], dtype=np.float32)).item()
    # similarity_2 = similarity_1.item()

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
        predicted_img = {
            'predicted_img_id': '',
            'bison_id': bison_id,
            'high_similarity': similarity_2,
            'low_similarity': similarity_1
        }
        print ("Similarity is equal")
    return predicted_img


compute_vector_represetation(
'/Users/yesidcano/repos/binary-image-selection/annotations/bison_annotations.cocoval2014.json',
'/Users/yesidcano/Downloads/val2014/')
