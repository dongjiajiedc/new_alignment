import numpy as np
import torch
import torch.utils.data as data

class balance_dataset(data.Dataset):

    def __init__(self,similarities, num_samples,embeddings,distances,datas):
        """Creates Hierarchical Clustering dataset with triples.

        @param labels: ground truth labels
        @type labels: np.array of shape (n_datapoints,)
        @param similarities: pairwise similarities between datapoints
        @type similarities: np.array of shape (n_datapoints, n_datapoints)
        """
        self.similarities = similarities
        self.embeddings = embeddings
        self.distances = distances
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
        i = data[0][0]
        j = data[0][1]
        k = data[1]
        f = data[2]
        isleaf = data[3]
        father = data[4]
        
        # constance_distance = self.distances[f] + self.distances[k] - 2 * self.distances[lca]
        sim=[self.similarities[i,k],self.similarities[j,k]];
        return torch.tensor(i),torch.tensor(j),torch.tensor(f),torch.tensor(sim),torch.tensor(isleaf),torch.tensor(father);
    
        # for i in data[1]:
        #     sim.append(self.similarities[i[0],i[1]]);
        # return 
        # return torch.tensor(data[0][0]),torch.tensor(data[0][1]),torch.tesor(data[1]),torch.tensor(distance),torch.tensor(sim)
        # return torch.from_numpy(triple), torch.from_numpy(similarities)

    def generate(self, num_samples):
        temp = []
        for i in range(len(self.datas)):
            temp.append(self.datas[i]);
        temp = np.array(temp)
        # self.datas= temp;
        if(num_samples <0):
            num_samples = len(temp)
        elif num_samples <=len(temp):
            rnd = np.random.choice(np.arange(len(temp)), num_samples, replace=False)
            temp = temp[rnd]
        else:
            n_pairs = len(temp);
            k_base = int(num_samples / n_pairs)
            k_rem = num_samples - (k_base * n_pairs)
            subset = np.random.choice(np.arange(n_pairs), k_rem, replace=False)
            pairs_rem = temp[subset]
            pairs_base = np.repeat(np.expand_dims(temp, 0), k_base, axis=0).reshape((-1,len(temp[0])))
            temp = np.concatenate([pairs_base, pairs_rem], axis=0)
            
        self.datas=temp