# hyper

```python
def get_Hyper_tree(data_path,start,end,label,epoches,model_path=None,model_path2=None,save_path='./'):
    """
    Embedding the dataset into hyperbolic tree structure
    
    Parameters
    ----------
    data_path : string
        Path of the cluster center file
    start : int
        Index of the starting in the data file
    end : int
        Index of the ending in the data file
    label : int
        Index of the label in the data file
    epoches : int
        Number of epochs for hyper-embedding model of the first dataset
    model_path : string
        Path to the model of the embedding
    model_path2 : string
        Path to the model of the rotating
    save_path : string 
        Path to the folder to save the data files

    Returns
    -------
    Save the training models and tree structure files in the folder

    """
```

```python
def train(model,dataloader,optimizer,similarities,epoches):
    """
    Train the embedding model
    """


```

```python
def train2(model,dataloader,optimizer,epoches):
    """
    Train the rotation model
    """

```

```python
def deep_search_tree(now,depth,path,f):
    """
    Search the tree and calculate the information
    """

```

```python

def search_merge_tree(now,ids,save_path,values,fathers,xys):
    """
    Search the tree and save the information
    """
```

```python

def get_colors(y, color_seed=1234):
    """
    Random color assignment for label classes.
    """
```

```python

def complete_tree(tree, leaves_embeddings):
    """
    Get embeddings of internal nodes from leaves' embeddings using LCA construction.
    """
```

```python

def is_leaf(tree, node):
    """
    Check if node is a leaf in tree.
    """
```

```python

def hyp_lca_numpy(x, y):
    """
    Computes the hyperbolic LCA in numpy.
    """
```

```python

def sl_np_mst_ij(xs, S):
    """
    Return the ij to merge the unionfind
    """
```
