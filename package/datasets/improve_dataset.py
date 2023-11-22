import numpy as np
import torch
import torch.utils.data as data

class IMDataset(data.Dataset):

    def __init__(self,similarities, num_samples,leaves_embeddings,datas):
        """Creates Hierarchical Clustering dataset with triples.

        @param labels: ground truth labels
        @type labels: np.array of shape (n_datapoints,)
        @param similarities: pairwise similarities between datapoints
        @type similarities: np.array of shape (n_datapoints, n_datapoints)
        """
        self.similarities = similarities
        self.leaves_embeddings = leaves_embeddings
        self.n_nodes = self.similarities.shape[0] -1
        self.datas = datas
        self.num_samples= num_samples
        self.generate(num_samples)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        
        # s12 = self.similarities[triple[0], triple[1]]
        # s13 = self.similarities[triple[0], triple[2]]
        # s23 = self.similarities[triple[1], triple[2]]
        # similarities = np.array([s12, s13, s23])
        sim=[];
        for i in data[1]:
            sim.append(self.similarities[i[0],i[1]]);

        return torch.tensor(data[0]),torch.tensor(data[1]),torch.tensor(sim)
        # return torch.from_numpy(triple), torch.from_numpy(similarities)

    def generate(self, num_samples):
        temp = []
        for i in range(len(self.datas)):
            data_i = self.datas[i];
            for j in data_i:
                temp.append([i,j]);
        temp = np.array(temp)
        # self.datas= temp;
        
        
        if num_samples > len(temp):
            n_pairs = len(temp);
            k_base = int(num_samples / n_pairs)
            k_rem = num_samples - (k_base * n_pairs)
            subset = np.random.choice(np.arange(n_pairs), k_rem, replace=False)
            pairs_rem = temp[subset]
            pairs_base = np.repeat(np.expand_dims(temp, 0), k_base, axis=0).reshape((-1, 2))
            temp = np.concatenate([pairs_base, pairs_rem], axis=0)
            
        self.datas=temp