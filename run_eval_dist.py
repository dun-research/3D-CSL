import argparse
import os
import os.path as osp
import warnings
from time import time
import mmcv
import torch
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist
from mmaction.datasets import build_dataloader
from mmaction.utils import (build_ddp, build_dp, default_device,
                            setup_multi_processes)
from mmaction.apis.test import collect_results_cpu, collect_results_gpu

try:
    from mmcv.engine import multi_gpu_test, single_gpu_test
except (ImportError, ModuleNotFoundError):
    warnings.warn(
        'DeprecationWarning: single_gpu_test, multi_gpu_test, '
        'collect_results_cpu, collect_results_gpu from mmaction2 will be '
        'deprecated. Please install mmcv through master branch.')
    from mmaction.apis import multi_gpu_test, single_gpu_test
from models.model import SimilarityRecognizer
from datasets.eval_dataset import build_eval_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Eval model on distributed environment')

    parser.add_argument('--weights', help='checkpoint file')
    parser.add_argument("--model", type=str, choices=['base', 'small', ], default='base')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')

    
    parser.add_argument("--batch_size_test", type=int, default=16)
    parser.add_argument("--dataset", type=str, choices=['fivr-5k', 'fivr-200k', 'cc_web_video'], default='fivr-5k')
    parser.add_argument("--num_workers_of_writer", type=int, default=4)
    parser.add_argument("--topk-cs", default=False, action="store_true")
    parser.add_argument(
                    '--tmpdir',
                            help='tmp directory used for collecting results from multiple '
                                    'workers, available when gpu-collect is not specified')
    parser.add_argument(
                    '--gpu-collect',
                            action='store_true',
                                    help='whether to use gpu to collect results')
    parser.add_argument(
        '-la', '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument(
        "-lcd", '--load_computed',
        action='store_true',
        help='load precomputed results')

    args = parser.parse_args()

    return args


def build_model(model_type, weight_path, batch_size_test=16):
    model = SimilarityRecognizer(model_type, batch_size_test)
    model.load_pretrained_weights(weight_path)
    model.eval()
    model = model.cuda()
    return model


def multi_gpu_test_dev(  # noqa: F811
            model, data_loader, tmpdir=None, gpu_collect=True):
        """Test model with multiple gpus.

        This method tests model with multiple gpus and collects the results
        under two different modes: gpu and cpu modes. By setting
        'gpu_collect=True' it encodes results to gpu tensors and use gpu
        communication for results collection. On cpu mode it saves the results
        on different gpus to 'tmpdir' and collects them by the rank 0 worker.

        Args:
            model (nn.Module): Model to be tested.
            data_loader (nn.Dataloader): Pytorch data loader.
            tmpdir (str): Path of directory to save the temporary results from
                different gpus under cpu mode. Default: None
            gpu_collect (bool): Option to use either gpu or cpu to collect
                results. Default: True

        Returns:
            list: The prediction results.
        """
        model.eval()
        results = []
        dataset = data_loader.dataset
        rank, world_size = get_dist_info()
        if rank == 0:
            prog_bar = mmcv.ProgressBar(len(dataset))
        for frames, video_id in data_loader:
            frames = frames[0]
            with torch.no_grad():
                result = model(frames)
                result = result.detach().cpu().numpy()
            results.append(result)

            if rank == 0:
                # use the first key as main key to calculate the batch size
                batch_size = 1
                for _ in range(batch_size * world_size):
                    prog_bar.update()

        rank, _ = get_dist_info()
        # collect results from all ranks
        if  gpu_collect:
            results = collect_results_gpu(results, len(dataset))
        else:
            results = collect_results_cpu(results, len(dataset), tmpdir)
        return results

def inference_pytorch(args, cfg, distributed, data_loader):
    """Get predictions by pytorch models."""

    # build the model and load checkpoint
    model = build_model(args.model, args.weights, args.batch_size_test)

    print("distributed: {}".format(distributed))
    start = time()
    if not distributed:
        model = build_dp(
            model, default_device, default_args=dict(device_ids=cfg.get("gpu_ids", [0,])))
        outputs = single_gpu_test(model, data_loader)

    else:
        model = build_ddp(
            model,
            default_device,
            default_args=dict(
                device_ids=[int(os.environ['LOCAL_RANK'])],
                broadcast_buffers=False))
        outputs = multi_gpu_test_dev(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
    end = time()
    print("Inference dataset use {}s".format(int(end - start)))
    return outputs


def setup_env(args):

    cfg_dict = dict(
        dist_params = dict(backend='nccl'),
        data=dict(
                videos_per_gpu=1,
                workers_per_gpu=4,
        )
    )
    cfg = Config(cfg_dict=cfg_dict)

    # set multi-process settings
    setup_multi_processes(cfg)


    # load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # overwrite output_config from args.out
        out_file = osp.join(args.out, "prediction.pkl")
        output_config = Config._merge_a_into_b(
            dict(out=out_file), output_config)
        os.makedirs(args.out, exist_ok=True)

    # load eval_config from cfg
    eval_config = cfg.get('eval_config', {})

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" ')


    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])
    cfg.distributed = distributed

    print("config: \n{}".format(cfg))
    print("args: \n{}".format(args))
    return cfg, output_config, eval_config


def compute_similarities(q_feat, d_feat, topk_cs=True):
        sim = q_feat @ d_feat.T
        sim = sim.max(dim=1)[0]
        if topk_cs:
            sim = sim.sort()[0][-3:]
        sim = sim.mean().item()
        return sim


def main():
    args = parse_args()
    cfg, output_config, eval_config = setup_env(args)

    # build the dataloader
    dataset = build_eval_dataset(args.dataset)

    print("videos_per_gpu: {}".format(cfg.data.get('videos_per_gpu', 1),))
    print("workers_per_gpu: {}".format(cfg.data.get('workers_per_gpu', 1),))

    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=cfg.distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    print(dataloader_setting)
    data_loader = build_dataloader(dataset, **dataloader_setting)

    print("data loader size: {}".format(len(data_loader)))

    if args.load_computed:
        outputs = mmcv.load(output_config['out'])
    else:
        outputs = inference_pytorch(args, cfg, cfg.distributed, data_loader)

    rank, _ = get_dist_info()
    if rank == 0:
        if output_config.get('out', None) and not args.load_computed:
            out = output_config['out']
            print(f'\nwriting results to {out}')
            mmcv.dump(outputs, file=out)

        all_videos = dataset.queries_ids + dataset.database_ids
        dim = outputs[0].shape[-1]
        all_features = {vid: torch.from_numpy(feat).cuda().reshape(-1, dim) for vid, feat in zip(all_videos, outputs)}
        all_features = {vid: feats / feats.norm(dim=-1, keepdim=True) for vid, feats in all_features.items()}
        similarities = {}

        for q_id in dataset.queries_ids:
            query_feat = all_features[q_id]
            similarities[q_id] = {}

            for d_id in dataset.database_ids:
                db_feat = all_features[d_id]
                sim_score = compute_similarities(query_feat, db_feat, args.topk_cs)
                similarities[q_id][d_id] = sim_score

        eval_res = dataset.evaluate(similarities, **eval_config)
        for name, val in eval_res.items():
            print(f'{name}: {val:.04f}')
    else:
        print(f"exit rank {rank}")

if __name__ == '__main__':
    main()

