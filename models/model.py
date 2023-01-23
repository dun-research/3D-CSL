
import torch
from torch import  nn
from .backbone import Timesformer


class SimilarityRecognizer(nn.Module):
    def __init__(self, model_type, batch_size=8):
        super().__init__()
        assert model_type in ['base', 'small']
        self.feature_extractor = Timesformer(img_size=224, patch_size=16, num_frames=8, vit_type=model_type)
        out_dim = 384 if model_type == "small" else 768
        self.batch_size = batch_size

    def load_pretrained_weights(self, filename):
        if isinstance(filename, str):
            checkpoint = torch.load(filename, map_location="cpu")
        
        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                f'No state_dict found in checkpoint file {filename}')
        # get state_dict from checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=True)
    def forward(self, imgs):
        return self.extract_features(imgs)

    def extract_features(self, imgs):
        """
            imgs shape: (batch_size, n_frames, n_channels, height, width)
        """
        max_bs = self.batch_size
        n_imgs = imgs.size(0)
        st = 0
        all_feats = []

        if n_imgs < max_bs:
            return self.feature_extractor.extract_features(imgs)

        while st < n_imgs:
            cur_imgs = imgs[st : st+max_bs]
            feats = self.feature_extractor.extract_features(cur_imgs)
            all_feats.append(feats)
            st += max_bs
        all_feats = torch.cat(all_feats, dim=0)
        
        return all_feats

    def compute_similarities(self, q_feat, d_feat, topk_cs=True):
        sim = q_feat @ d_feat.T
        sim = sim.max(dim=1)[0]
        if topk_cs:
            sim = sim.sort()[0][-3:]
        sim = sim.mean().item()
        return sim
        
    def normalize_features(self, feats):
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats
        
