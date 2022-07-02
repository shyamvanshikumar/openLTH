from cmath import nan
import json
import math
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
from scipy import sparse
import scipy.linalg
from foundations import paths
from platforms.platform import get_platform

def density(G):
    top = {n for n,d in G.nodes(data=True) if d["bipartite"] == 0}
    return bipartite.density(G, top)

def clustering_coeff(G):
    clust_coeff = bipartite.clustering(G)
    avg_coeff = sum(clust_coeff.values()) / len(clust_coeff)
    return {"clustering coefficients": clust_coeff, "average coefficient": avg_coeff}

def centrality(G, type="degree"):
    top = {n for n,d in G.nodes(data=True) if d["bipartite"] == 0}
    if type == "closeness":
        return bipartite.closeness_centrality(G, top)
    elif type == "degree":
        return bipartite.degree_centrality(G, top)
    elif type == "betweenness":
        return bipartite.betweenness_centrality(G, top)
    else:
        return ValueError('No such centrality measure exists')

def redundancy_coeff(G):
    try:
        rc = bipartite.node_redundancy(G)
        avg_rc = sum(rc.values()) / len(G)

        return {"node redundancy": rc, "average redundancy": avg_rc}
    except nx.exception.NetworkXError:
        return "Redundancy calculation not possible due to a node with degree less than 2"

def degree_statistics(G):
    top = {n for n,d in G.nodes(data=True) if d["bipartite"] == 0}
    node_degrees = bipartite.degrees(G, top)
    degOut, degIn = dict(node_degrees[0]), dict(node_degrees[1])
    maxIn, minIn, avgIn = max(degIn.values()), min(degIn.values()), sum(degIn.values()) / len(degIn)
    maxOut, minOut, avgOut = max(degOut.values()), min(degOut.values()), sum(degOut.values()) / len(degOut)

    Out = {"deg dist": degOut, "max degree": maxOut, "min degree": minOut, "avg degree": avgOut}
    In = {"deg dist": degIn, "max degree": maxIn, "min degree": minIn, "avg degree": avgIn}

    return Out, In

def spectral_statistics(G):
    top = {n for n,d in G.nodes(data=True) if d["bipartite"] == 0}
    eigenvalues =  scipy.linalg.eigvals(nx.adjacency_matrix(G).todense()).real

    eigenvalues = np.sort(eigenvalues)
    t1, t2 = eigenvalues[-1], eigenvalues[-2]

    try:
        deltaS = (2 * math.sqrt(t1 - 1) - t2) / t2
    except ValueError:
        deltaS = "eigenvalues less than 1 therefore complex deltaS"
    
    left, right = degree_statistics(G)
    davgLeft = left["avg degree"]
    davgRight = right["avg degree"]

    try:
        deltaR = (math.sqrt(davgLeft - 1) + math.sqrt(davgRight - 1) - t2) / t2
    except ValueError:
        deltaR = "average degrees less than 1"
        
    specGap = t1 - t2
    return {"t1": t1, "t2": t2, "delta S": deltaS, "delta R": deltaR, "t1 - t2": specGap}

def add_mask_metrics(G):
        layer_metrics = {}
        layer_metrics["Clustering"] = clustering_coeff(G)
        layer_metrics["Redundancy"] = redundancy_coeff(G)
        layer_metrics["Density"] = density(G)
        layer_metrics["Centrality"] = centrality(G)
        layer_metrics["degree statistics"] = degree_statistics(G)
        layer_metrics["Spectral statistics"] = spectral_statistics(G)

        return layer_metrics

def add_weighted_metrics(G):
    weighted_metrics = {}
    weighted_metrics["Weighted spectral statistics"] = spectral_statistics(G)
    return weighted_metrics


class GraphMetricLogger:
    def __init__(self, location):
        self.log = {}
        self.location = location

    def eval_graph_metrics(self, model):
        for name, param in model.named_parameters():
            if "bias" in name:
                continue

            bi_adj = param.to(float).detach().numpy()
            if "bias" in name:
                bi_adj = np.absolute(bi_adj)
            bi_adj = sparse.csr_matrix(bi_adj)
            G = bipartite.from_biadjacency_matrix(bi_adj)

            if "mask" in name:
                self.log[name] = add_mask_metrics(G)
            elif "weight" in name:
                self.log[name] = add_weighted_metrics(G)
    
    def save(self):
        if not get_platform().is_primary_process: return
        if not get_platform().exists(self.location):
            get_platform().makedirs(self.location)
        with get_platform().open(paths.metricLogger(self.location), 'w') as fp:
            json.dump(self.log, fp)