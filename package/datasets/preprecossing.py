from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
import scanpy.external as sce

def preprocessing(
    adata: AnnData,
    Min_Genes: int = 200,
    Min_Cells: int = 3,
    Min_Mean: float = 0.0125,
    Max_Mean: float = 3,
    Min_Disp: float = 0.5,
    N_pcs: int = 50,
    n_Top_genes: int = 2000,
    K: int = 10,
    magic_imputation: bool = False,
):
    """\
    The recipe for preprocessing raw count data.

    In adata.raw, all genes are stored, so that those genes can be used in later calculation in CAPITAL.

    The recipe runs the following steps:
    ::

        import scanpy as sc
        sc.pp.filter_cells(adata, min_genes=Min_Genes)
        sc.pp.filter_genes(adata, min_cells=Min_Cells)
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, min_mean=Min_Mean, max_mean=Max_Mean, min_disp=Min_Disp, n_top_genes=n_Top_genes)
        adata.raw = adata
        adata = adata[:,adata.var['highly_variable']]
        sc.tl.pca(adata, n_comps=N_pcs)
        sc.pp.neighbors(adata, n_neighbors=K, n_pcs=N_pcs)
        sc.tl.diffmap(adata)
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        sc.tl.paga(adata, groups='leiden')

    Parameters
    ----------
    adata : AnnData
        The annotated data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    Min_Genes : int
        The number of genes to filter in scanpy.pp.filter_cells(),
        by default 200.
    Min_Cells : int
        The number of cells to filter in scanpy.pp.filter_genes(),
        by default 3.
    Min_Mean : float
        The minimum mean that is filtered to calculate highly variable genes.
        Look scanpy.pp.highly_variable_genes(),
        by default 0.0125.
    Max_Mean : int
        The maxmum mean that is filtered to calculate highly variable genes.
        Look scanpy.pp.highly_variable_genes(),
        by default 3.
    Min_Disp : float
        The minimum dispersion that is filtered to calculate highly variable genes.
        Look scanpy.pp.highly_variable_genes(),
        by default 0.5.
    N_pcs : int
        The number of principal components used,
        by default 50.
    n_Top_genes : int
        The number of highly variable genes,
        by default 2000.
    K : int
        The size of a local neighborhood used for manifold approximation,
        by default 10.
    magic_imputation : bool
        If `True`, MAGIC imputation is done,
        by default `False`.
    """

    if not isinstance(adata, AnnData):
        raise ValueError("preprocessing() expects AnnData argument")

    pp = Preprocessing()
    pp.preprocessing_rawdata(
        adata,
        Min_Genes=Min_Genes,
        Min_Cells=Min_Cells,
        Min_Mean=Min_Mean,
        Max_Mean=Max_Mean,
        Min_Disp=Min_Disp,
        N_pcs=N_pcs,
        n_Top_genes=n_Top_genes,
        K=K,
        magic_imputation=magic_imputation,
        copy=False
    )

class Preprocessing:
    def __init__(self):
        pass

    def preprocessing_rawdata(
        self,
        adata,
        Min_Genes=200,
        Min_Cells=3,
        Min_Mean=0.0125,
        Max_Mean=3,
        Min_Disp=0.5,
        N_pcs=50,
        n_Top_genes=2000,
        K=10,
        magic_imputation=False,
        copy=False
    ):

        # adata = adata.copy() if copy else adata
        sc.pp.filter_cells(adata, min_genes=Min_Genes)
        sc.pp.filter_genes(adata, min_cells=Min_Cells)
        sc.pp.normalize_total(adata, exclude_highly_expressed=True)
        sc.pp.log1p(adata)

        if magic_imputation is True:
            #     set np.random.seed for magic to create same imputation
            np.random.seed(1234)
            sce.pp.magic(
                adata,
                name_list='all_genes',
                knn=5
            )

        sc.pp.highly_variable_genes(
            adata,
            min_mean=Min_Mean,
            max_mean=Max_Mean,
            min_disp=Min_Disp,
            n_top_genes=n_Top_genes
        )
        adata.raw = adata
        # same as adata = adata[:,adata.var['highly_variable']] but inplace
        adata._inplace_subset_var(adata.var['highly_variable'])
        sc.tl.pca(
            adata,
            n_comps=N_pcs
        )
        sc.pp.neighbors(adata,
                        n_neighbors=K,
                        n_pcs=N_pcs,
                        random_state=1234
                        )
        sc.tl.diffmap(adata,random_state=1234)
        sc.tl.umap(adata,random_state=1234)
        sc.tl.leiden(adata,random_state=1234)
        sc.tl.paga(adata, groups='leiden')

        return adata if copy else None

    # cluster_centroid: pd.DataFrame
    # index is cluster name, columns is gene name, X is gene expression level
    # argument "dimension" is either pca or diffmap
def preprocessing_cluster(adata,
                        N_pcs=50,
                        K=10,
                        copy=False,
                        resolution=0.5,
                    ):
        adata.raw = adata

        # adata._inplace_subset_var(adata.var['highly_variable'])

        sc.tl.pca(
            adata,
            n_comps=N_pcs,
            random_state=1234,
        )
        sc.pp.neighbors(adata,
                        n_neighbors=K,
                        n_pcs=N_pcs,
                        random_state=1234
                        )
        sc.tl.diffmap(adata,random_state=1234)
        sc.tl.umap(adata,random_state=1234)
        sc.tl.leiden(adata,random_state=1234,resolution = resolution)
        sc.tl.paga(adata, groups='leiden')

        return adata if copy else None
def preprocessing_st_cluster(adata,
                        N_pcs=20,
                        K=10,
                        copy=False,
                        resolution=0.5,
                    ):
        adata.raw = adata

        # adata._inplace_subset_var(adata.var['highly_variable'])

        # sc.tl.pca(
        #     adata,
        #     n_comps=N_pcs
        # )
        sc.pp.neighbors(adata,
                        n_neighbors=K,
                        # n_pcs=N_pcs,
                        random_state=1234
                        )
        sc.tl.diffmap(adata,random_state=1234)
        sc.tl.umap(adata,random_state=1234)
        sc.tl.leiden(adata,random_state=1234,resolution = resolution)
        sc.tl.paga(adata, groups='leiden')

        return adata if copy else None
    
def calculate_cluster_centroid(
    adata,
    dimension="pca",
    groupby="leiden"
):

    if dimension == "pca":
        X_dimension = "X_pca"
    elif dimension == "diffmap":
        X_dimension = "X_diffmap"
    elif dimension == "raw":
        X_dimension = "raw"
    else:
        raise ValueError(
            "Argument 'dimension' must be 'pca' or 'diffmap'.")

    if dimension in ["pca", "diffmap"]:
        clustername = adata.obs[groupby].cat.categories
        cluster_centroid_data = np.empty(
            (0, adata.obsm[X_dimension].shape[1]))
        for i in clustername:
            a_cluster_data = pd.DataFrame(
                adata[adata.obs[groupby] == "{}".format(i)].obsm[X_dimension])
            a_cluster_median = a_cluster_data.median(axis=0).values
            cluster_centroid_data = np.vstack(
                (cluster_centroid_data, a_cluster_median))
    else:
        clustername = adata.obs[groupby].cat.categories
        cluster_centroid_data = np.empty(
            (0, adata.X.shape[1]))
        for i in clustername:
            a_cluster_data = pd.DataFrame(
                adata[adata.obs[groupby] == "{}".format(i)].X)
            a_cluster_median = a_cluster_data.median(axis=0).values
            cluster_centroid_data = np.vstack(
                (cluster_centroid_data, a_cluster_median))

    return cluster_centroid_data

def set_initial_condition(
        adata,
        groupby="leiden",
        method="euclid",
        dimension="pca",
        copy=False
    ):
    if not isinstance(adata.X, np.ndarray):
        adata_tmp = adata.copy()
        adata_tmp.X = adata.X.toarray()

    else:
        adata_tmp = adata.copy()
        
    cluster_centroid_data = calculate_cluster_centroid(
                        adata_tmp,
                        dimension=dimension,
                        groupby=groupby
                    )
    adata.uns["cluster_centroid"] = cluster_centroid_data
    adata.uns["capital"] = {}
    adata.uns["capital"]["tree"] = {}
    tree_dict = adata.uns["capital"]["tree"]
    tree_dict["annotation"] = groupby
    return adata if copy else None

def calculate_cluster_centroid_for_genes(
        adata,
        gene_list,
        save_path="./",
    ):
    groupby = adata.uns["capital"]["tree"]["annotation"]
    filtered_data = adata.raw.to_adata()[:, gene_list]
    # filtered_data.to_df().to_csv(save_path+"data_cell.csv");
    # adata.obs.to_csv(save_path+"data_type.csv")
    cluster_centroid_data = np.empty((0, filtered_data.n_vars))
    clustername = filtered_data.obs[groupby].unique().tolist()

    for i in clustername:
        a_cluster_data = filtered_data[filtered_data.obs[groupby] == "{}".format(
            i)].to_df()
        a_cluster_median = a_cluster_data.mean(axis=0).values
        cluster_centroid_data = np.vstack(
            (cluster_centroid_data, a_cluster_median)
        )
    clustername = list(map(int, clustername))
    
    cluster_centroid = pd.DataFrame(
        cluster_centroid_data,
        index=clustername,
        columns=filtered_data.var_names
    ).sort_index()
    cluster_centroid.to_csv(save_path+"datas.data",header=None);
    cluster_centroid.to_csv(save_path+"datas.csv");

    return cluster_centroid

def sort_data(
    adata1,
    adata2,
    N_1=2000,
    N_2=2000
):
    if N_1 is not None:
        adata1 = adata1.raw.to_adata()
        sc.pp.highly_variable_genes(adata1, n_top_genes=N_1,flavor='seurat_v3')
        adata1 = adata1[:, adata1.var['highly_variable']]
    elif N_1 is None:
        pass

    if N_2 is not None:
        adata2 = adata2.raw.to_adata()
        sc.pp.highly_variable_genes(adata2, n_top_genes=N_2,flavor='seurat_v3')
        adata2 = adata2[:, adata2.var['highly_variable']]
    elif N_2 is None:
        pass

    s1 = set(adata1.var.index)
    s2 = set(adata2.var.index)
    intersection_list = list(s1.intersection(s2))

    if len(intersection_list) < 2:
        raise ValueError("highly variable genes of intersection of data1 and data2 are not enough "\
                            "to calculate the cost of a tree alignment. \n"\
                            "Specify num_genes1 and num_genes2 carefully.")

    print("{} genes are used to calculate cost of tree alignment.\n".format(
        len(intersection_list)))

    return intersection_list