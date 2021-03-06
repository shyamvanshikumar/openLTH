import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from networkx.algorithms import bipartite
from sparse.graph_generators import registry

from foundations import hparams
from lottery.desc import LotteryDesc
from sparse.models import base
from pruning import sparse_global
from sparse.models import sparse_linear as sl


class Model(base.Model):
    '''A LeNet sparsely-connected model for CIFAR-10'''

    def __init__(self, plan, generator, density, initializer, outputs=10):
        super(Model, self).__init__()

        layers = []
        current_size = 784  # 28 * 28 = number of pixels in MNIST image.
        for i, size in enumerate(plan):
            G = generator(current_size, size, density[i])
            mask = bipartite.biadjacency_matrix(G, row_order=list(range(current_size))).toarray()
            layers.append(sl.SparseLinear(current_size, size, mask))
            current_size = size

        self.fc_layers = nn.ModuleList(layers)
        G = generator(current_size, outputs, density[-1])
        mask = bipartite.biadjacency_matrix(G, row_order=list(range(current_size))).toarray()
        self.fc = sl.SparseLinear(current_size, outputs, mask)
        self.criterion = nn.CrossEntropyLoss()

        self.apply(initializer)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten.
        for layer in self.fc_layers:
            x = F.relu(layer(x))

        return self.fc(x)

    @property
    def output_layer_names(self):
        return ['fc.weight', 'fc.bias']

    @staticmethod
    def is_valid_model_name(model_name):
        return (model_name.startswith('mnist_sparse_lenet') and
                len(model_name.split('_')) > 2 and
                all([x.isdigit() and int(x) > 0 for x in model_name.split('_')[3:]]))

    @staticmethod
    def get_model_from_name(model_name, initializer, density, gen_name, outputs=None):
        """The name of a model is mnist_lenet_N1[_N2...].

        N1, N2, etc. are the number of neurons in each fully-connected layer excluding the
        output layer (10 neurons by default). A LeNet with 300 neurons in the first hidden layer,
        100 neurons in the second hidden layer, and 10 output neurons is 'mnist_lenet_300_100'.
        """

        outputs = outputs or 10

        if not Model.is_valid_model_name(model_name):
            raise ValueError('Invalid model name: {}'.format(model_name))

        plan = [int(n) for n in model_name.split('_')[3:]]
        generator = registry.get(gen_name)

        density = [float(d) for d in density.split(',')]
        if(len(density) != len(plan)+1):
            raise ValueError('Number of density values is not compatible with number of trainable layers')
        return Model(plan, generator, density, initializer, outputs)

    @property
    def loss_criterion(self):
        return self.criterion

    @staticmethod
    def default_hparams():
        model_hparams = hparams.ModelHparams(
            model_name='mnist_sparse_lenet_300_100',
            model_init='kaiming_normal',
            batchnorm_init='uniform'
        )

        dataset_hparams = hparams.DatasetHparams(
            dataset_name='mnist',
            batch_size=128
        )

        training_hparams = hparams.TrainingHparams(
            optimizer_name='sgd',
            lr=0.1,
            training_steps='40ep'
        )

        sparse_hparams = hparams.SparseHparams(
            density=0.3,
            graph_generator='Erdos'
        )

        pruning_hparams = sparse_global.PruningHparams(
            pruning_strategy='sparse_global',
            pruning_fraction=0.2,
            pruning_layers_to_ignore='fc.weight',
        )

        return LotteryDesc(model_hparams, dataset_hparams, training_hparams, pruning_hparams, sparse_hparams)