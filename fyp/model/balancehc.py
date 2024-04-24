import torch
import torch.nn as nn
import torch.nn.functional as F
from alignment import *

from utils.poincare import hyp_dist
import numpy as np

class balancehc(nn.Module):


    def __init__(self,nodes,embeddings,hyperparamter=1, rank=2, temperature=0.05, init_size=1e-3, max_scale=1. - 1e-3,):
        
        super(balancehc, self).__init__()
        self.nodes = nodes
        self.leaves_embeddings = embeddings
        self.n_nodes = len(embeddings)
        self.embeddings = nn.Embedding(self.n_nodes, rank)
        self.temperature = temperature
        self.scale = nn.Parameter(torch.Tensor([init_size]), requires_grad=True)
        self.embeddings.weight.data = torch.tensor(embeddings);
        self.init_size = init_size
        self.hyperparamter = hyperparamter
        self.max_scale = max_scale
        self.original = np.array(self.leaves_embeddings)
        self.distances = [i.distance_to_root for i in self.nodes];

    def anneal_temperature(self, anneal_factor):
        self.temperature *= anneal_factor

    def normalize_embeddings(self, embeddings):
        min_scale = 1e-2 #self.init_size
        max_scale = self.max_scale
        return F.normalize(embeddings, p=2, dim=1) * self.scale.clamp_min(min_scale).clamp_max(max_scale)
    
    def loss(self,i,j,k,sims,isleaf,f):
        if(isleaf==1):
            e1 = self.embeddings(i);
            e2 = self.embeddings(j);
            e3 = self.embeddings(k).clone().detach();
            # e1 = self.normalize_embeddings(e1);
            # e2 = self.normalize_embeddings(e2);
            # e3 = self.normalize_embeddings(e3);
            
            origin_distance = hyp_dist(torch.tensor(self.original[i]),torch.tensor(self.original[j]));
            lca = int(self.find_lca(self.nodes[f],self.nodes[k]))
            constant_distance = self.distances[f] + self.distances[k] - 2 * self.distances[lca]

            d1 = hyp_dist(e1,e3) + constant_distance;
            d2 = hyp_dist(e2,e3) + constant_distance;
            lca_norm = torch.cat([d1, d2], dim=-1)
            weights = torch.softmax(lca_norm / self.temperature, dim=-1)
            w_ord = torch.sum(sims * weights, dim=-1, keepdim=True)
            total = torch.sum(sims, dim=-1, keepdim=True) - w_ord + self.hyperparamter * torch.abs(origin_distance - hyp_dist(e1,e2));
            return torch.mean(total)
        else:

            # e1 = self.normalize_embeddings(e1);
            # e2 = self.normalize_embeddings(e2);
            # e3 = self.normalize_embeddings(e3);
            node1 = self.nodes[i];
            node2 = self.nodes[j];
            nodef = self.nodes[k];
            
            e1 = self.embeddings(torch.tensor(int(nodef.son[0]))).clone().detach();
            e2 = self.embeddings(torch.tensor(int(nodef.son[1]))).clone().detach();
            e3 = self.embeddings(k);
            e4 = self.embeddings(f).clone().detach();

            origin_distance = hyp_dist(torch.tensor(self.original[int(nodef.son[0])]),torch.tensor(self.original[k])) + hyp_dist(torch.tensor(self.original[int(nodef.son[1])]),torch.tensor(self.original[k]));
            lca = int(self.find_lca(self.nodes[f],self.nodes[k]))
            constant_distance = self.distances[f] + self.distances[k] - 2 * self.distances[lca]
            
            d1 = hyp_dist(e1,e3) + hyp_dist(e3,e4) + constant_distance  + node1.distance_to_root;
            d2 = hyp_dist(e2,e3) + hyp_dist(e3,e4) + constant_distance  + node2.distance_to_root;

            if i in nodef.son[0].subson:
                d1 = d1 + nodef.son[0].distance_to_root
            else:
                d1 = d1 + nodef.son[1].distance_to_root
                
            if j in nodef.son[0].subson:
                d2 = d2 + nodef.son[0].distance_to_root
            else:
                d2 = d2 + nodef.son[1].distance_to_root     
                           

            lca_norm = torch.cat([d1, d2], dim=-1)
            weights = torch.softmax(lca_norm / self.temperature, dim=-1)
            w_ord = torch.sum(sims * weights, dim=-1, keepdim=True)
            total = torch.sum(sims, dim=-1, keepdim=True) - w_ord + self.hyperparamter*torch.abs(origin_distance - hyp_dist(torch.tensor(self.original[int(nodef.son[0])]),e3) - hyp_dist(torch.tensor(self.original[int(nodef.son[1])]),e3));
            return torch.mean(total)   
    def deep_search_tree(self,now,depth,path,f):
        now.f=f
        now.depth=depth;
        path.append(now);
        now.path=path.copy();
        if(f!=now):
            now.distance_to_root = f.distance_to_root + hyp_dist(f.value,now.value)
        else:
            now.distance_to_root = 0
        
        for i in now.son:
            self.deep_search_tree(i,depth+1,path,now);
        path.remove(now)
    
    def find_lca(self,node1,node2):
        minnum = min(len(node1.path),len(node2.path));
        for i in range(minnum):
            if(node1.path[i]!=node2.path[i]):
                return node1.path[i-1];
        return node1.path[minnum-1];
    
    def update_node(self):
        for i in range(len(self.nodes)):
            self.nodes[i].value = self.original[i]
        pass;
    def update(self):
        self.original = self.embeddings.weight.data
        self.update_node();
        self.deep_search_tree(self.nodes[-1],0,[],self.nodes[-1])
        self.distances = [i.distance_to_root for i in self.nodes];

