from read_data import *
from hyper import *
from alignment import *
from datasets.preprecossing import *
def run(data1_path,data2_path,folder1_path,folder2_path):
    # adata1 = sc.read("../../../capital/docs/tutorials/BRCA_EMTAB8107_expression_processed.h5ad")
    # adata2 = sc.read("../../../capital/docs/tutorials/BRCA_GSE114727_inDrop_expression_processed.h5ad") 
    adata1 = sc.read(data1_path)
    adata2 = sc.read(data2_path)
    # preprocessing()
    # sc.pl.umap(adata1, color="leiden")
    set_initial_condition(adata1)
    set_initial_condition(adata2)
    adata2.uns.pop("log1p")
    adata1.uns.pop("log1p")
    gene_list = sort_data(adata1,adata2)
    adata1.uns["capital"]["intersection_genes"] = np.array(
    gene_list, dtype=object)
    adata2.uns["capital"]["intersection_genes"] = np.array(
        gene_list, dtype=object)
    datas1 = calculate_cluster_centroid_for_genes(adata1,gene_list,save_path = folder1_path)
    datas2 = calculate_cluster_centroid_for_genes(adata2,gene_list,save_path = folder2_path)
    get_Hyper_tree(folder1_path+'datas.data',1,514,0,10,save_path=folder1_path)
    get_Hyper_tree(folder2_path+'datas.data',1,514,0,10,save_path=folder2_path)
    run_alignment(folder_path1=folder1_path,folder_path2=folder1_path)
if __name__ == "__main__":
    run("../../../capital/docs/tutorials/BRCA_EMTAB8107_expression_processed.h5ad","../../../capital/docs/tutorials/BRCA_GSE114727_inDrop_expression_processed.h5ad","./datas/data1/",'./datas/data1/')
