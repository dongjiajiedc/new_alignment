"""Hyperbolic hierarchical clustering model."""

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.poincare import hyp_dist


class improvehc(nn.Module):
    """
    Hyperbolic embedding model for hierarchical clustering.
    """

    def __init__(self,leaves_embeddings,dumpy_node, n_nodes=1, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3):
        super(improvehc, self).__init__()
        self.leaves_embeddings = leaves_embeddings
        self.n_nodes = n_nodes
        self.embeddings = nn.Embedding(n_nodes, rank)
        self.temperature = temperature
        self.scale = nn.Parameter(torch.Tensor([init_size]), requires_grad=True)
        # self.embeddings.weight.data = project(
        #     self.scale * (2 * torch.rand((n_nodes, rank)) - 1.0)
        # )
        self.embeddings.weight.data = torch.tensor(dumpy_node);
        self.init_size = init_size
        self.max_scale = max_scale

    def anneal_temperature(self, anneal_factor):
        """

        @param anneal_factor: scalar for temperature decay
        @type anneal_factor: float
        """
        self.temperature *= anneal_factor

    def normalize_embeddings(self, embeddings):
        """Normalize leaves embeddings to have the lie on a diameter."""
        min_scale = 1e-2 #self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)

    # def loss(self, triple_ids, similarities):
    #     """Computes the HypHC loss.
    #     Args:
    #         triple_ids: B x 3 tensor with triple ids
    #         similarities: B x 3 tensor with pairwise similarities for triples 
    #                       [s12, s13, s23]
    #     """
    #     e1 = self.embeddings(triple_ids[:, 0])
    #     e2 = self.embeddings(triple_ids[:, 1])
    #     e3 = self.embeddings(triple_ids[:, 2])
    #     e1 = self.normalize_embeddings(e1)
    #     e2 = self.normalize_embeddings(e2)
    #     e3 = self.normalize_embeddings(e3)
    #     d_12 = hyp_lca(e1, e2, return_coord=False)
    #     d_13 = hyp_lca(e1, e3, return_coord=False)
    #     d_23 = hyp_lca(e2, e3, return_coord=False)
    #     lca_norm = torch.cat([d_12, d_13, d_23], dim=-1)
    #     weights = torch.softmax(lca_norm / self.temperature, dim=-1)
    #     w_ord = torch.sum(similarities * weights, dim=-1, keepdim=True)
    #     total = torch.sum(similarities, dim=-1, keepdim=True) - w_ord
    #     return torch.mean(total)
    def loss(self,lca,l_and_r,sims):
        e1 = self.embeddings(lca);
        e1=self.normalize_embeddings(e1)
        distance = [];
        for l,r in l_and_r[0]:
            d1 = self.leaves_embeddings[l]
            d2 = self.leaves_embeddings[r]
            d3 = hyp_dist(e1,d1) + hyp_dist(e1,d2)
            distance.append(d3)

        distance=torch.stack(distance).reshape(-1)
        weights = torch.softmax(distance / self.temperature, dim=-1)
        w_ord = torch.sum(sims * weights, dim=-1, keepdim=True)
        total = torch.sum(sims, dim=-1, keepdim=True) - w_ord
        return torch.mean(total)
                
        
