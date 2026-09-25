import os
import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback


class SetStaticGraph(Callback):
    """Enable DDP static-graph (torch 1.10 has no ctor kwarg -> call the method before the first
    forward). Needed for GSF, which reuses stereo_volume_encoder for the color+gray volumes and,
    with gradient checkpointing (with_cp), otherwise triggers DDP 'mark ready once' / hang."""
    def on_train_start(self, trainer, pl_module):
        m = getattr(trainer.strategy, 'model', None)
        if m is not None and hasattr(m, '_set_static_graph'):
            m._set_static_graph()
            print('[DDP] static graph ENABLED')


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/semantic_kitti.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--resume', action='store_true')   # resume from the latest last.ckpt

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    cfg.update(vars(args))
    return args, cfg

if __name__ == '__main__':
    args, config = parse_config()
    log_folder = os.path.join('logs', config['log_folder'])
    misc.check_path(log_folder)

    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    model = pl_model(config)
    
    data_dm = DataModule(config)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/mIoU',
        mode='max',
        save_top_k=3,            # best 3 by mIoU (save_top_k=-1 = every epoch filled the 1.8T disk -> crash)
        save_last=True,          # rolling last.ckpt for resume
        filename='epoch{epoch:02d}')   # no '/' in name -> avoids the nested-dir bug
    
    if not config.eval:
        resume_ckpt = None
        if config.get('resume'):
            import glob as _glob
            cands = _glob.glob(os.path.join('logs', config['log_folder'], 'tensorboard',
                                            'version_*', 'checkpoints', 'last.ckpt'))
            if cands:
                resume_ckpt = max(cands, key=os.path.getmtime)
                print(f'[resume] resuming from {resume_ckpt}')
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            max_steps=config.training_steps,
            num_sanity_val_steps=0,
            resume_from_checkpoint=resume_ckpt,
            limit_train_batches=config.get('limit_train_batches', 1.0),   # small-batch validation runs
            callbacks=[
                checkpoint_callback,
                LearningRateMonitor(logging_interval='step')
            ],
            logger=tb_logger,
            profiler=profiler,
            sync_batchnorm=True,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config['check_val_every_n_epoch']
        )
        if resume_ckpt is None and config['ckpt_path'] is not None:
            # fine-tune INIT from the released CGFormer ckpt (strict=False: fusion is new + zero-init).
            # (skipped when resuming -> resume_from_checkpoint restores full state incl. optimizer.)
            ckpt = torch.load(config['ckpt_path'], map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f'[finetune init strict=False] missing={len(missing)} unexpected={len(unexpected)}')
            print('  missing (should be only model.fusion.*):', list(missing)[:12])
        trainer.fit(model=model, datamodule=data_dm)
    else:
        trainer = pl.Trainer(
            devices=[i for i in range(num_gpu)],
            strategy=DDPStrategy(
                accelerator='gpu',
                find_unused_parameters=False
            ),
            limit_test_batches=config.get('limit_test_batches', 1.0),  # subset quick-eval (default: full)
            logger=tb_logger,
            profiler=profiler
        )
        if config['ckpt_path'] is not None:
            # strict=False: the fusion module is NEW (not in the released CGFormer ckpt); it is
            # zero-init so the model still starts EXACTLY at the color-only baseline.
            ckpt = torch.load(config['ckpt_path'], map_location='cpu')
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f'[load strict=False] missing={len(missing)} unexpected={len(unexpected)}')
            print('  missing sample:', list(missing)[:10])
            print('  unexpected sample:', list(unexpected)[:10])
            trainer.test(model=model, datamodule=data_dm)
        else:
            trainer.test(model=model, datamodule=data_dm)

    

