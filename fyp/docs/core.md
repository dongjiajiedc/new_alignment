# core

```python
def merge_by_radius(cell_path,folder_path,radius,method='average',meta_col='celltype'):
    """
    Merge the cells of the datasets according to the radius 
    
    Parameters
    ----------
    cell_path : string
        Path to the dataset's cell data h5ad file 
    folder_path1 : string
        Path to the folder to save the result files of the dataset      
    radius : float
        Radius for merging cells in the dataset
    method : string
        Method for merging cells
        'average' for using average value
        'median' for using median value
        'max' for using max value
    meta_col : string
        The column name which contain the information of celltype
        
    Returns
    -------
    Save the result merged files in the folder and return the loss of the merging according to the meta information
    """
```

```python
def alignment_process(cell_path1,cell_path2,folder_path1,folder_path2,radius1,radius2,c1,c2,epoches1,epoches2,meta_col='celltype',contin=True,resolution=0.5,method='average',alignment=1,n_pca=50):
    """
    Performs alignment of two datasets. 
    
    Parameters
    ----------
    cell_path1 : string
        Path to the first dataset's cell data h5ad file 
    cell_path2 : string
        Path to the second dataset's cell data h5ad file 
    folder_path1 : string
        Path to the folder to save the files of the first dataset
    folder_path2 : string
        Path to the folder to save the files of the second dataset        
    radius1 : float
        Radius for merging cells in the first dataset
    radius2 : float
        Radius for merging cells in the second dataset
    c1 : float
        Parameter for the merging tree node for first dataset
    c2 : float
        Parameter for the merging tree node for first dataset
    epoches1 : int
        Number of epochs for hyper-embedding model of the first dataset
    epoches2 : int
        Number of epochs for hyper-embedding model of the second dataset
    meta_col : string
        The column name which contain the information of celltype
    contin : boolean 
        Boolean flag indicating whether to continue from previous alignment data files
    resolution : float 
        Resolution parameter for clustering
    method : string
        Method for merging cells
        'average' for using average value
        'median' for using median value
        'max' for using max value
    alignment : int
        Alignment method to use 
        1 for dp algorithm
        2 for linear programming
    n_pca : int
        Parameter of PCA for the clustering
        
    Returns
    -------
    Returns the alignment correctness 

    """
```