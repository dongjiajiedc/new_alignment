from read_data import *
from hyper import *
from alignment import *
from datasets.preprecossing import *
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score


def alignment_process(cell_path1,cell_path2,celltype_path1,celltype_path2,celltype_column1,celltype_column2,folder_path1,folder_path2,radius1,radius2,c1,c2,epoches1,epoches2):
    loss = merge_cells_by_radius(cell_path1,celltype_path1,celltype_column1,folder_path1,radius1)
    print("cell merge loss for dataset1: {}".format(loss))

    loss = merge_cells_by_radius(cell_path2,celltype_path2,celltype_column2,folder_path2,radius2)
    print("cell merge loss for dataset2: {}".format(loss))


    adata1 = pd.read_csv(folder_path1+"merge_cell_data.csv")
    cell_meta = pd.read_csv(folder_path1+"merge_cell_meta.csv")
    cell_meta = cell_meta.set_index(cell_meta.columns[0])
    adata1 = adata1.set_index(adata1.columns[0])
    adata1 = anndata.AnnData(adata1)
    adata1.obs['celltype'] = cell_meta.values

    adata2 = pd.read_csv(folder_path2+"merge_cell_data.csv")
    cell_meta = pd.read_csv(folder_path2+"merge_cell_meta.csv")
    cell_meta = cell_meta.set_index(cell_meta.columns[0])
    adata2 = adata2.set_index(adata2.columns[0])
    adata2 = anndata.AnnData(adata2)
    adata2.obs['celltype'] = cell_meta.values

    preprocessing_cluster(adata1,N_pcs=50)
    preprocessing_cluster(adata2,N_pcs=50)
    # preprocessing_rawdata(adata1,N_pcs=50)
    # preprocessing_rawdata(adata2,N_pcs=50)
    # preprocessing_rawdata
    set_initial_condition(adata1)
    set_initial_condition(adata2)
    inter_gene = sort_data(adata1,adata2)
    tmp1 = calculate_cluster_centroid_for_genes(adata1,inter_gene,folder_path1)
    tmp2 = calculate_cluster_centroid_for_genes(adata2,inter_gene,folder_path2)
    ari = adjusted_rand_score(adata1.obs['celltype'].tolist(), adata1.obs['leiden'].tolist())
    print("ARI score for adata1: ", ari)
    ari = adjusted_rand_score(adata2.obs['celltype'].tolist(), adata2.obs['leiden'].tolist())
    print("ARI score for adata2: ", ari)
    get_Hyper_tree(adata1,folder_path1+'datas.data',1,tmp1.shape[1]+1,0,epoches1,model_path=None,save_path=folder_path1,c=0)
    get_Hyper_tree(adata2,folder_path2+'datas.data',1,tmp2.shape[1]+1,0,epoches2,model_path=None,save_path=folder_path2,c=0)
        
    nodes1,n1 = build_hyper_tree(folder_path1)
    show_tree(nodes1[0]).show_fig()
    nodes2,n2 = build_hyper_tree(folder_path2)
    show_tree(nodes2[0]).show_fig()
    T=tree_alignment(nodes1[0],nodes2[0],1);
    minn = T.run_alignment();
    T.show_ans();
    ans = T.get_ans()
    G=show_graph(ans,nodes1[0],nodes2[0]);
    G.show_fig()
    print("average cost for one node:{}".format(minn/(n1+n2)))
    
def alignment_process_st(cell_path1,cell_path2,celltype_path1,celltype_path2,celltype_column1,celltype_column2,folder_path1,folder_path2,radius1,radius2,c1,c2,epoches1,epoches2):
        
    merge_st_by_radius(cell_path1,celltype_path1,celltype_column1,folder_path1,radius1)
    
    adata1 = pd.read_csv(folder_path1+"merge_cell_data.csv")
    cell_meta = pd.read_csv(folder_path1+"merge_cell_meta.csv")
    cell_meta = cell_meta.set_index(cell_meta.columns[0])
    adata1 = adata1.set_index(adata1.columns[0])
    adata1 = anndata.AnnData(adata1)
    adata1.obs['celltype'] = cell_meta.values
    
    merge_st_by_radius(cell_path2,celltype_path2,celltype_column2,folder_path2,radius2)

    adata2 = pd.read_csv(folder_path2+"merge_cell_data.csv")
    cell_meta = pd.read_csv(folder_path2+"merge_cell_meta.csv")
    cell_meta = cell_meta.set_index(cell_meta.columns[0])
    adata2 = adata2.set_index(adata2.columns[0])
    adata2 = anndata.AnnData(adata2)
    adata2.obs['celltype'] = cell_meta.values
    

    
    preprocessing_cluster(adata1,N_pcs=1)
    preprocessing_cluster(adata2,N_pcs=1)

    set_initial_condition(adata1)
    set_initial_condition(adata2)

    inter_gene = sort_data(adata1,adata2)

    tmp1 = calculate_cluster_centroid_for_genes(adata1,inter_gene,folder_path1)
    tmp2 = calculate_cluster_centroid_for_genes(adata2,inter_gene,folder_path2)
    get_Hyper_tree(folder_path1+'datas.data',1,tmp1.shape[1]+1,0,epoches1,model_path=None,save_path=folder_path1,c=0)
    get_Hyper_tree(folder_path2+'datas.data',1,tmp2.shape[1]+1,0,epoches2,model_path=None,save_path=folder_path2,c=0)
        
    nodes1,n1 = build_hyper_tree(folder_path1)
    nodes2,n2 = build_hyper_tree(folder_path2)

    merge_list1 = [];
    merge_list2 = [];
    nodes1[0] = search_tree(nodes1[0],c1,merge_list1)
    nodes2[0] = search_tree(nodes2[0],c2,merge_list2)


    T=tree_alignment(nodes1[0],nodes2[0],1);
    minn = T.run_alignment();
    T.show_ans();
    ans = T.get_ans()
    G=show_graph(ans,nodes1[0],nodes2[0]);
    G.show_fig()
    print("average cost for one node:{}".format(minn/(n1+n2)))
