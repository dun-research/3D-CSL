import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch

from models.model import SimilarityRecognizer
from datasets.eval_dataset import build_eval_dataset


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", 
                        type=str, 
                        choices=['base', 'small', ], 
                        default='base',
                        help="Model architecture used for evaluation ")
    parser.add_argument("--weights", 
                        type=str, 
                        help="Pretrained weights ")
    parser.add_argument("--dataset", 
                        type=str, 
                        choices=['fivr-5k', 'fivr-200k', 'cc_web_video'], 
                        default='fivr-5k', 
                        help="The dataset used for evaluation")
    parser.add_argument("--batch_size_test", 
                        type=int, 
                        default=16,
                        help="Batch size per gpu when tested")
    parser.add_argument("--out_file", 
                        type=str, 
                        help="File where the results will be stored")
    parser.add_argument("--topk-cs", 
                        default=False, 
                        action="store_true",
                        help="Flag that indicated whether use topk chamfer similarities as described in our paper")
    args = parser.parse_args()
    return args


def build_model(model_type, weight_path, batch_size_test=16):
    model = SimilarityRecognizer(model_type, batch_size_test)
    model.load_pretrained_weights(weight_path)
    model.eval()
    model = model.cuda()
    return model


if __name__ == "__main__":
    args = args_parser()
    
    model = build_model(args.model, args.weights, args.batch_size_test)
    dataset = build_eval_dataset(args.dataset)
    
    all_features = {}
    print("Starting to extract features....")
    for frames, video_id in tqdm(dataset):
        frames = frames.cuda()
        with torch.no_grad():
            feats = model.extract_features(frames)
        all_features[video_id] = feats.detach().cpu()
    

    print("Starting to compute similarities....")
    normed_features = {dataset.get_uid_by_idx(vid): model.normalize_features(feats.cuda()) for vid, feats in all_features.items()}
    similarities = {}

    for q_id in dataset.queries_ids:
        query_feat = normed_features[q_id]
        similarities[q_id] = {}

        for d_id in dataset.database_ids:
            db_feat = normed_features[d_id]
            sim_score = model.compute_similarities(query_feat, db_feat, args.topk_cs)
            similarities[q_id][d_id] = sim_score
    
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    json.dump(similarities, open(args.out_file, "w"))
            

    print("Starting to evaluate....")    
    eva_res = dataset.evaluate(similarities)



