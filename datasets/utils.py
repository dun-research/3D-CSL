""" 
    # Copyright (c) MKLab-ITI. All Rights Reserved.
    # Code is modified from https://github.com/MKLab-ITI/visil/blob/master/datasets/__init__.py
"""

import os.path as osp
import numpy as np
import pickle


class CC_WEB_VIDEO():
    def __init__(self, ann_file):
        with open(ann_file, 'rb') as f:
           dataset = pickle.load(f)
        self.database = dataset['index']
        self.queries = dataset['queries']
        self.ground_truth = dataset['ground_truth']
        self.excluded = dataset['excluded']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(map(str, self.database.keys()))

    def calculate_mAP(self, similarities, all_videos=False, clean=False, positive_labels='ESLMV'):
        mAP = 0.0
        missing_videos =  {'8/8_271_Y.avi', '6/6_563_Y.avi', '6/6_498_Y.avi', '4/4_160_Y.wmv', '6/6_526_Y.avi'}
        for query_set, labels in enumerate(self.ground_truth):
            query_id = self.queries[query_set]
            i, ri, s = 0.0, 0.0, 0.0
            if query_id in similarities:
                res = similarities[query_id]
                for video_id in sorted(res.keys(), key=lambda x: res[x], reverse=True):
                    video = self.database[video_id]
                    if (all_videos or video in labels) and (not clean or video not in self.excluded[query_set]):
                        if video in missing_videos:
                            continue
                        ri += 1
                        if video in labels and labels[video] in positive_labels:
                            i += 1.0
                            s += i / ri
                            if ri != i:
                                print(f"error: {i}/{ri}/score{round(res[video_id],3)}, {video_id}")
                positives = np.sum([1.0 for k, v in labels.items() if
                                    v in positive_labels and (not clean or k not in self.excluded[query_set])])
                mAP += s / positives
            print(f"query: {query_id} ,map={round(s/positives,3)}")
        return mAP / len(set(self.queries).intersection(similarities.keys()))

    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        print('=' * 5, 'CC_WEB_VIDEO Dataset', '=' * 5)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))
        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))

        print('-' * 25)
        print('All dataset')

        CC_WEB_mAP = self.calculate_mAP(similarities, all_videos=False, clean=False)
        CC_WEB_mAP_x = self.calculate_mAP(similarities, all_videos=True, clean=False)
        CC_WEB_mAP_clean = self.calculate_mAP(similarities, all_videos=False, clean=True)
        CC_WEB_mAP_x_clean = self.calculate_mAP(similarities, all_videos=True, clean=True)
        
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}\n'.format(
            CC_WEB_mAP,CC_WEB_mAP_x))

        print('Clean dataset')
        print('CC_WEB mAP: {:.4f}\nCC_WEB* mAP: {:.4f}'.format(
            CC_WEB_mAP_clean,CC_WEB_mAP_x_clean))
        
        return {
            "CC_WEB_mAP": CC_WEB_mAP,
            "CC_WEB_mAP_x":CC_WEB_mAP_x, 
            "CC_WEB_mAP_clean":CC_WEB_mAP_clean,
            "CC_WEB_mAP_x_clean":CC_WEB_mAP_x_clean,
        }



class FIVR(object):
    def __init__(self, ann_file, version='5k', ):
        self.version = version
        with open(ann_file, 'rb') as f:
            dataset = pickle.load(f)
        self.annotation = dataset['annotation']
        self.queries = dataset[self.version]['queries']
        self.database = dataset[self.version]['database']

    def get_queries(self):
        return self.queries

    def get_database(self):
        return list(self.database)

    def calculate_mAP(self, query, res, all_db, relevant_labels):
        gt_sets = self.annotation[query]
        query_gt = set(sum([gt_sets[label] for label in relevant_labels if label in gt_sets], []))
        query_gt = query_gt.intersection(all_db)
        if len(query_gt)==0:
            return 1.0
        i, ri, s = 0.0, 0, 0.0

        cnt =0
        
        keys = set(res.keys()).intersection(all_db)
        sort_res = sorted(keys, key=lambda x: res[x], reverse=True)
        
        sort_res = np.array(list(sort_res))
        sort_res = sort_res[sort_res != query]
        
        for video in sort_res:
            ri += 1
            if video in query_gt:
                i += 1.0
                cnt  += 1
                s += i / ri
            if  cnt >= len(query_gt): 
                break
        return s / len(query_gt)


    def evaluate(self, similarities, all_db=None):
        if all_db is None:
            all_db = self.database

        DSVR, CSVR, ISVR = [], [], []
        for query, res in similarities.items():
            if query in self.queries:
                map = self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS'])
                DSVR.append(map)
                CSVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS']))
                ISVR.append(self.calculate_mAP(query, res, all_db, relevant_labels=['ND', 'DS', 'CS', 'IS']))
             

        msg = f"======FIVR-{self.version.upper()} Dataset ========="
        print(msg)
        not_found = len(set(self.queries) - similarities.keys())
        if not_found > 0:
            print('[WARNING] {} queries are missing from the results and will be ignored'.format(not_found))

        print('Queries: {} videos'.format(len(similarities)))
        print('Database: {} videos'.format(len(all_db)))
        print('-' * 16)
        print('DSVR mAP: {:.4f}'.format(np.mean(DSVR)))
        print('CSVR mAP: {:.4f}'.format(np.mean(CSVR)))
        print('ISVR mAP: {:.4f}'.format(np.mean(ISVR)))
        return {"DSVR": np.mean(DSVR), "CSVR": np.mean(CSVR), "ISVR":np.mean(ISVR), }



