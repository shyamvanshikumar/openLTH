import argparse
from dataclasses import dataclass

from cli import arg_utils
from cli import shared_args
from foundations.runner import Runner
import sparse.models.registry
from platforms.platform import get_platform
from training import train
from sparse.desc import SparseDesc
from sparse.metric_logger import GraphMetricLogger

@dataclass
class SparseRunner(Runner):
    replicate: int
    desc: SparseDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return "Train a sparse model."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        shared_args.JobArgs.add_args(parser)
        SparseDesc.add_args(parser, SparseRunner.maybe_get_default_hparams())

    @staticmethod
    def maybe_get_default_hparams():
        default_hparams = arg_utils.maybe_get_arg('default_hparams')
        return sparse.models.registry.get_default_hparams(default_hparams) if default_hparams else None

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'SparseRunner':
        return SparseRunner(args.replicate, SparseDesc.create_from_args(args),
                              not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate))

    def run(self):
        if self.verbose and get_platform().is_primary_process:
            print('='*82 + f'\nTraining a Model (Replicate {self.replicate})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.replicate)}' + '\n' + '='*82 + '\n')
        self.desc.save(self.desc.run_path(self.replicate))
        model = sparse.models.registry.get(self.desc.model_hparams, self.desc.sparse_hparams, self.desc.run_path(self.replicate))
        train.standard_train(
            model, self.desc.run_path(self.replicate), self.desc.dataset_hparams, 
            self.desc.training_hparams, evaluate_every_epoch=self.evaluate_every_epoch)
        
        logger = GraphMetricLogger(self.desc.run_path(self.replicate))
        logger.eval_graph_metrics(model)
        logger.save()