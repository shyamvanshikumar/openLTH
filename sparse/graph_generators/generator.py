import abc

import networkx as nx
from networkx.algorithms import bipartite

class Graphs(abc.ABC):

    @staticmethod
    @abc.abstractclassmethod
    def is_valid_name(gen_name: str):
        pass

    @staticmethod
    @abc.abstractclassmethod
    def get_generator_from_name(gen_name: str):
        pass

class Erdos_Reyni(Graphs):

    @staticmethod
    def is_valid_name(gen_name):
        return (gen_name.lower() == "erdos")
    
    @staticmethod
    def get_generator_from_name(gen_name):
        return bipartite.random_graph
        
