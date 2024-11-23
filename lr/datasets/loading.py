"""Dataset loading."""

import os

import numpy as np



def load_data(dataset, start_idx,end_idx,label_idx,normalize=True):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    x = []
    y = []
    data_path = dataset
    classes = {}
    class_counter = 0
    with open(data_path, 'r') as f:
        for line in f:
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    # print(x);
    y = np.array(y, dtype=int) 
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    std = x.std(0)
    x = (x - mean) / std
    
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x0 = x[None, :, :]
    x1 = x[:, None, :]
    cos = (x0 * x1).sum(-1)
    similarities = 0.5 * (1 + cos)
    similarities = np.triu(similarities) + np.triu(similarities).T
    similarities[np.diag_indices_from(similarities)] = 1.0
    similarities[similarities > 1.0] = 1.0
    return x, y, similarities
