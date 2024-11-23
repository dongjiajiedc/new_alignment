from hyper import *
from alignment import *
from eval import *
from datasets.preprecossing import *
from sklearn.metrics import adjusted_rand_score
from utils import *
from scipy.spatial import KDTree
from alignment import *
from tqdm import tqdm
import os
import calendar
import time
import pandas as pd
import sys
import anndata
import scanpy as sc

def add_meta(now,meta_list,merge):
    if(int(now.name)<len(meta_list)):
        now.name= now.name +'_'+ meta_list[int(now.name)];
    merge.append(now)
    for i in now.son:
        add_meta(i,meta_list,merge)
def remove_meta(now):
    if(len(now.name.split('_')) >1):
        now.name = now.name.split('_')[0]
    for i in now.son:
        remove_meta(i)
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
    
    # np.random.seed(1234)
    datas = sc.read_h5ad(cell_path)
    sc.pp.filter_cells(datas, min_genes=200)
    sc.pp.filter_genes(datas, min_cells=3)
    sc.pp.normalize_total(datas, exclude_highly_expressed=True)
    sc.pp.log1p(datas)
    sc.pp.highly_variable_genes(
        datas,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        n_top_genes=2000
    )
    datas.raw = datas
    datas._inplace_subset_var(datas.var['highly_variable'])

    celltype = datas.obs[meta_col]
    index_names = datas.to_df().index.tolist()
    datas = datas.to_df()
    adata = datas.copy()
    ans_value = []
    ans_label = []
    ori_index = []
    true_label = [];
    now_labels = list(celltype)
    r= radius
    now_label = 0;
    now =  datas.values;
    progress_bar = tqdm(total=len(now), ncols=80)

    if(r <=0):
        ans_value = now
        ans_label = [i for i in range(len(ans_value))]
        true_label = list(celltype)
        ori_index = index_names;
        progress_bar.update(len(now))

    else:
        while len(now) != 0:
            rnd = np.random.randint(now.shape[0], size=1);
            rand_choice = now[rnd, :].reshape(-1)
            tree = KDTree(now);
            indices = tree.query_ball_point(rand_choice,r)
            now = now.tolist();

            for i in indices:
                ans_value.append(now[i])
                ans_label.append(now_label);
                true_label.append(now_labels[i]);
                ori_index.append(index_names[i]);
            now_label+=1;
            indices.sort(reverse=True);
            for i in indices:
                now.pop(i)
                now_labels.pop(i)
                index_names.pop(i)
            
            now = np.array(now);
            progress_bar.update(len(indices))
            sys.stdout.flush()

    progress_bar.close()


    v = pd.DataFrame(ans_value)
    v['meta_label'] = ans_label
    if(method=='average'):
        ann  = v.groupby("meta_label").mean()
    elif(method=='median'):
        ann = v.groupby("meta_label").median()
    elif(method=='max'):
        ann = v.groupby("meta_label").max()

    ann.columns = datas.columns


    v1 = pd.DataFrame(true_label)
    v1['meta_label'] = ans_label
    meta = v1.groupby("meta_label").max()
    loss1 = 1-(v1.groupby("meta_label").describe()[0]['count'] - v1.groupby("meta_label").describe()[0]['freq']).sum() / (v1.groupby("meta_label").describe()[0]['count'].sum())
    v1['ori_index'] = ori_index
    v1.to_csv(folder_path +'merge_labels.csv');
    
    adata = sc.AnnData(ann);
    adata.obs['celltype']= meta.values.reshape(-1)
    adata.write_h5ad(folder_path + 'adata.h5ad');
    
    # return loss1

def calculate_cluster_celltype(adata,groupby = 'leiden', meta_col = 'celltype'):
    meta_list = []
    clustername = adata.obs[groupby].unique().tolist()
    clustername = list(map(int, clustername))
    clustername.sort()
    for value in clustername:
        indices = [i for i, x in enumerate(adata.obs[groupby]) if x == str(value)]
        t = [adata.obs[meta_col].tolist()[index] for index in indices]
        most_common_element = max(t, key=t.count)
        meta_list.append(most_common_element)
    return meta_list

def calculate_meta_ori(folder_path,adata):
    v = pd.read_csv(folder_path+"merge_labels.csv")
    meta = adata.obs['celltype']
    cell_type = []
    cluster = []
    for i in range(len(v)):
        cell_type.append(meta.iloc[v['meta_label'][i]][0])
        cluster.append(adata.obs['leiden'].iloc[v['meta_label'][i]][0])
    v['celltype_meta']=cell_type
    v['cluster']=cluster
    v.to_csv(folder_path+'meta_result.csv',index=None)
    
def alignment_process(cell_path1,cell_path2,folder_path1,folder_path2,radius1,radius2,c1,c2,epoches1,epoches2,meta_col='celltype',contin=False,resolution=0.5,method='average',alignment=1,n_pca=50,mst1=False):
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
    
    # current_GMT = time.gmtime()
    # ts = calendar.timegm(current_GMT)
    # print("Current timestamp:", ts)
    
    # log1 = open(folder_path1+"log_{}.txt".format(ts), "w")   
    # log2 = open(folder_path2+"log_{}.txt".format(ts), "w")
    # log1.write("args for data1: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path1,folder_path1,radius1,c1,epoches1))
    # log1.write("args for data2: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path2,folder_path2,radius2,c2,epoches2))
    # log2.write("args for data1: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path1,folder_path1,radius1,c1,epoches1))
    # log2.write("args for data2: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path2,folder_path2,radius2,c2,epoches2))

    if (contin==False) or (os.path.exists(folder_path1 + 'adata.h5ad') == False):
        loss1 = merge_by_radius(cell_path1,folder_path1,radius1,method,meta_col)
        print("cell meta score for dataset1: {}\n".format(loss1))
    else:
        print("dataset1 find files and skip merging")

    if(contin==False) or ( os.path.exists(folder_path2 + 'adata.h5ad') == False):
        loss2 = merge_by_radius(cell_path2,folder_path2,radius2,method,meta_col)
        print("cell meta score for dataset2: {}".format(loss2))
    else:
        print("dataset2 find files and skip merging")

    adata1 = sc.read_h5ad(folder_path1+ 'adata.h5ad')
    adata2 = sc.read_h5ad(folder_path2+ 'adata.h5ad')

    preprocessing_cluster(adata1,N_pcs=n_pca,resolution=resolution)
    preprocessing_cluster(adata2,N_pcs=n_pca,resolution=resolution)
    
    inter_gene = sort_data(adata1,adata2)

    tmp1 = calculate_cluster_centroid_for_genes(adata1,inter_gene,folder_path1)
    tmp2 = calculate_cluster_centroid_for_genes(adata2,inter_gene,folder_path2)
    
    # ari = adjusted_rand_score(adata1.obs['celltype'].tolist(), adata1.obs['leiden'].tolist())
    # print("ARI score for adata1: ", ari)
    # log1.write("ARI score for adata1: "+ str(ari)+'\n')
    
    # ari = adjusted_rand_score(adata2.obs['celltype'].tolist(), adata2.obs['leiden'].tolist())
    # print("ARI score for adata2: ", ari)
    # log2.write("ARI score for adata2: "+ str(ari)+'\n')
    
    meta_list1 = calculate_cluster_celltype(adata1);
    meta_list2 = calculate_cluster_celltype(adata2);
    calculate_meta_ori(folder_path1,adata1);
    calculate_meta_ori(folder_path2,adata2);
    
    if(contin==False) or ((os.path.exists(folder_path1 + 'dataxy.npy') and os.path.exists(folder_path1+'datalink.npy') and os.path.exists(folder_path1+'dataname.npy')) == False):
        embeddings1,nodes1 = get_Hyper_tree(folder_path1+'datas.data',1,tmp1.shape[1]+1,0,epoches1,10,save_path=folder_path1,mst1=mst1)
        merge_points_with_c(embeddings1,nodes1,folder_path1 +'datas.data',1,tmp1.shape[1]+1,0,folder_path1,epoches2,c1,c2)
        nos1 = build_hyper_tree_from_folder(folder_path1,True)
        add_meta(nos1[0],meta_list1,[])
        show_tree(nos1[0],color=['#184e77','#1a759f','#168aad',"#34a0a4",'#52b69a','#99d98c','#76c893','#99d98c']).show_fig()
        remove_meta(nos1[0]);
    else:
        print("dataset1 find files and skip embedding");

    if(contin==False) or ((os.path.exists(folder_path2 + 'dataxy.npy') and os.path.exists(folder_path2+'datalink.npy') and os.path.exists(folder_path2+'dataname.npy')) == False):
        embeddings2,nodes2 = get_Hyper_tree(folder_path2 +'datas.data',1,tmp2.shape[1]+1,0,epoches1,epoches2,meta_list2,save_path=folder_path2)
        merge_points_with_c(embeddings2,nodes2,folder_path2 +'datas.data',1,tmp2.shape[1]+1,0,folder_path2,epoches2,c1,c2)
        nos2 = build_hyper_tree_from_folder(folder_path2,True)
        add_meta(nos2[0],meta_list2,[])
        show_tree(nos2[0],color=['#184e77','#1a759f','#168aad',"#34a0a4",'#52b69a','#99d98c','#76c893','#99d98c']).show_fig()
        remove_meta(nos2[0]);
    else:
        print("dataset2 find files and skip embedding")

        
    nodes1 = build_hyper_tree_from_folder(folder_path1)
    nodes2 = build_hyper_tree_from_folder(folder_path2)

    nodes_merge1 = [];
    nodes_merge2 = [];
    add_meta(nodes1[0],meta_list1,[])
    add_meta(nodes2[0],meta_list2,[])

            
    rate = 0;        
    if(alignment==1):
        rate,anslist,ans = run_alignment(nodes1,nodes2,folder_path1,folder_path2,meta_list1,meta_list2);
    elif(alignment==2):
        rate = run_alignment_linear(nodes1,nodes2);
    
        
    # log1.write("Alignment score: "+ str(rate)+'\n')
    # log2.write("Alignment score: "+ str(rate)+'\n')
    # log1.close()
    # log2.close()