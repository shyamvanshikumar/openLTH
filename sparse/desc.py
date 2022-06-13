import argparse
from dataclasses import dataclass
import os

from datasets import registry as datasets_registry
from foundations import desc
from foundations import hparams
from foundations.step import Step
from lottery.desc import LotteryDesc
from platforms.platform import get_platform

@dataclass
class SparseDesc(desc.Desc):
    """The hyperparameters necessary to describe a training run."""

    model_hparams: hparams.ModelHparams
    dataset_hparams: hparams.DatasetHparams
    training_hparams: hparams.TrainingHparams
    sparse_hparams: hparams.SparseHparams

    @staticmethod
    def name_prefix(): return 'sparseTrain'

    @staticmethod
    def add_args(parser: argparse.ArgumentParser, defaults: LotteryDesc = None):
        hparams.DatasetHparams.add_args(parser, defaults=defaults.dataset_hparams if defaults else None)
        hparams.ModelHparams.add_args(parser, defaults=defaults.model_hparams if defaults else None)
        hparams.TrainingHparams.add_args(parser, defaults=defaults.training_hparams if defaults else None)
        hparams.SparseHparams.add_args(parser, defaults=defaults.sparse_hparams if defaults else None)
    
    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SparseDesc':
        dataset_hparams = hparams.DatasetHparams.create_from_args(args)
        model_hparams = hparams.ModelHparams.create_from_args(args)
        training_hparams = hparams.TrainingHparams.create_from_args(args)
        sparse_hparams = hparams.SparseHparams.create_from_args(args)
        return SparseDesc(model_hparams, dataset_hparams, training_hparams, sparse_hparams)

    def run_path(self, replicate, experiment='main'):
        return os.path.join(get_platform().root, self.hashname, f'replicate_{replicate}', experiment)
        
    @property
    def display(self):
        return '\n'.join([self.dataset_hparams.display, self.model_hparams.display, self.training_hparams.display, self.sparse_hparams.display])