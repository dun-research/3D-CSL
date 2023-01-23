
import os.path as osp
import pickle as pk


import numpy as np
from .utils import FIVR, CC_WEB_VIDEO
from .preprocess import Preprocess
import torch



class EvalDataset():
    def __init__(self,  query_file,
                        database_file,
                        anno_file,
                        data_root=None,
                        clip_len=8,
                        out_size=224,
                        dataset_name="fivr",
                        version="5k",
                        ):
        assert dataset_name in ['fivr', 'cc_web_video']
        self.load_annotations(query_file, database_file)
        self.modality = "RGB"
        self.data_prefix = data_root
        self.preprocess = Preprocess(clip_len=clip_len, out_size=out_size, frame_interval=1)


        if dataset_name == "fivr":
            self.evaluator = FIVR(ann_file=anno_file, version=version, )
        elif dataset_name == "cc_web_video":
            self.evaluator = CC_WEB_VIDEO(ann_file=anno_file)
        else:
            raise NotImplemented

    def load_annotations(self, query_file, database_file):
        self.queries = np.loadtxt(query_file, dtype=str)
        self.queries = np.expand_dims(self.queries, axis=0) if self.queries.ndim == 1 else self.queries
        self.queries = self.queries.tolist()
        self.queries_ids = [x[0] for x in self.queries]
        
        self.database = np.loadtxt(database_file, dtype=str)
        self.database = np.expand_dims(self.database, axis=0) if self.database.ndim == 1 else self.database
        self.database = self.database.tolist()
        self.database_ids = [x[0] for x in self.database]

        self.video_infos = self.queries + self.database 
        self.all_db =  self.database_ids
        print(f"read {len(self.queries)} query, {len(self.database)} database")
        
    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Prepare the frames for testing given the index."""
        uid, video_path = self.video_infos[idx]
        if self.data_prefix is not None:
            video_path = osp.join(self.data_prefix, video_path)
            frames = self.preprocess(video_path)
        frames = torch.from_numpy(frames).permute(0, 4, 1, 2, 3).float() # expect  (n_clips, n_frames, n_channels, height, width)
        return frames, idx

    def get_uid_by_idx(self, idx):
        return self.video_infos[idx][0]
    
    def evaluate(self, similarities):
        eval_res = self.evaluator.evaluate(similarities, self.all_db)
        return eval_res


def build_eval_dataset(dataset_name):
    if dataset_name == "fivr-5k":
        query_file = "data/fivr-5k-queries.txt"
        database_file = "data/fivr-5k-database.txt"
        ann_file = "data/fivr.pkl"
        data_root = "/data1/public/FIVR-200K/"
        dataset_name = "fivr"
        version = "5k"
    elif dataset_name == "fivr-200k":
        query_file = "data/fivr-200k-queries.txt"
        database_file = "data/fivr-200k-database.txt"
        ann_file = "data/fivr.pkl"
        data_root = "/data1/public/FIVR-200K/"
        dataset_name = "fivr"
        version = "200k"
    elif dataset_name == "cc_web_video":
        query_file = "data/cc-web-queries-24.txt"
        database_file = "data/cc-web-database-13091.txt"
        ann_file = "data/cc_web_video.pkl"
        data_root = "/data1/public/CC_WEB_VIDEO/Videos"
        dataset_name = "cc_web_video"
        version = ""
    else:
        raise ValueError

    dataset = EvalDataset(query_file, database_file, ann_file, data_root, 
                            clip_len=8, out_size=224, 
                            dataset_name=dataset_name, version=version)
    return dataset
