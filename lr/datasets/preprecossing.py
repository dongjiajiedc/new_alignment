from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc

 
def preprocessing_cluster(adata,
                        N_pcs=50,
                        K=30,
                        copy=False,
                        resolution=0.5,
                    ):
    
        # if(N_pcs > len(adata.var)):
        #     N_pcs = len(adata.var)

        sc.tl.pca(
            adata,
            n_comps=N_pcs,
            random_state=1234,
        )
        sc.pp.neighbors(adata,
                        n_neighbors=K,
                        random_state=1234,
                        n_pcs=N_pcs,
                        )
        sc.tl.diffmap(adata,random_state=1234)
        sc.tl.umap(adata,random_state=1234)
        sc.tl.leiden(adata,random_state=1234,resolution = resolution)
        sc.tl.paga(adata, groups='leiden')

        return adata if copy else None

def calculate_cluster_centroid_for_genes(
        adata,
        gene_list,
        save_path="./",
        groupby ='leiden',
        X_dimension='X_pca',
    ):
    filtered_data = adata[:, gene_list]
    cluster_centroid_data = np.empty((0, filtered_data.n_vars))
    clustername = filtered_data.obs[groupby].unique().tolist()
    for i in clustername:
        a_cluster_data = filtered_data[filtered_data.obs[groupby] == "{}".format(i)].to_df()
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
    # print(save_path+"datas.csv")
    cluster_centroid.to_csv(save_path+"datas.csv");
    
    
    if(X_dimension!='X_pca'):
        cluster_centroid.to_csv(save_path+"datas.data",header=None);
    else:
        cluster_centroid_data = np.empty((0, adata.obsm[X_dimension].shape[1]))
        clustername = filtered_data.obs[groupby].unique().tolist()

        for i in clustername:
            a_cluster_data = pd.DataFrame(filtered_data[filtered_data.obs[groupby] == "{}".format(i)].obsm[X_dimension])

            a_cluster_median = a_cluster_data.mean(axis=0).values
            cluster_centroid_data = np.vstack(
                (cluster_centroid_data, a_cluster_median)
            )
            
        clustername = list(map(int, clustername))
        cluster_centroid = pd.DataFrame(
            cluster_centroid_data,
            index=clustername,
        ).sort_index()

        cluster_centroid.to_csv(save_path+"datas.data",header=None);
    
    return cluster_centroid


def sort_data(
    adata1,
    adata2,
    N_1=2000,
    N_2=2000
):
    temp1 = adata1.copy()
    sc.pp.highly_variable_genes(temp1, n_top_genes=N_1)
    temp1 = temp1[:, temp1.var['highly_variable']]

    temp2 = adata2.copy()
    sc.pp.highly_variable_genes(temp2, n_top_genes=N_2)
    temp2 = temp2[:, temp2.var['highly_variable']]
    
    s1 = set(temp1.var.index)
    s2 = set(temp2.var.index)
    intersection_list = list(s1.intersection(s2))

    return intersection_list