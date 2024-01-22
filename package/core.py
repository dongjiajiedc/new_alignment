from hyper import *
from alignment import *
from datasets.preprecossing import *
from sklearn.metrics import adjusted_rand_score
from utils import *
from scipy.spatial import KDTree
from alignment import *
from tqdm import tqdm

import calendar
import time
import pandas as pd
import sys
import anndata
import scanpy as sc


def merge_by_radius(cell_path,folder_path,radius,method='average',meta_col='celltype'):
    np.random.seed(1234)
    datas = sc.read_h5ad(cell_path)
    celltype = datas.obs[meta_col]
    datas = datas.to_df()
    adata = datas.copy()
    adata['Celltype']= list(celltype)
    # adata.loc[list(celltype['Cell']), 'Celltype'] = list(celltype[celltype_column])
    ans_value = []
    ans_label = []
    true_label = [];
    now_labels = adata['Celltype'].tolist();
    r= radius
    now_label = 0;
    now =  datas.values;

    progress_bar = tqdm(total=len(now), ncols=80)
    while len(now) != 0:
        rnd = np.random.randint(now.shape[0], size=1);
        rand_choice = now[rnd, :].reshape(-1)
        tree = KDTree(now);
        indices = tree.query_ball_point(rand_choice,r)
        points_within_k = now[indices]
        now = now.tolist();

        for i in points_within_k:
            ans_value.append(i.tolist())
            ans_label.append(now_label);
            index = now.index(i.tolist());
            true_label.append(now_labels[index]);
            now.pop(index)
            now_labels.pop(index)
        now_label+=1;
            
        now = np.array(now);
        progress_bar.update(len(points_within_k))
        sys.stdout.flush()

    progress_bar.close()

    v = pd.DataFrame(ans_value)
    v['label'] = ans_label
    if(method=='average'):
        ann  = v.groupby("label").mean()
    elif(method=='median'):
        ann = v.groupby("label").median()
    elif(method=='max'):
        ann = v.groupby("label").max()

    ann.columns = datas.columns


    v1 = pd.DataFrame(true_label)
    v1['label'] = ans_label
    meta = v1.groupby("label").max()
    loss1 = 1-(v1.groupby("label").describe()[0]['count'] - v1.groupby("label").describe()[0]['freq']).sum() / (v1.groupby("label").describe()[0]['count'].sum())
    v.to_csv(folder_path +'merge_values.csv');
    v1.to_csv(folder_path +'merge_labels.csv');
    ann.to_csv(folder_path + 'merge_cell_data.csv');
    meta.to_csv(folder_path + 'merge_cell_meta.csv');
    
    return loss1

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
    
    current_GMT = time.gmtime()
    ts = calendar.timegm(current_GMT)
    print("Current timestamp:", ts)
    
    log1 = open(folder_path1+"log_{}.txt".format(ts), "w")   
    log2 = open(folder_path2+"log_{}.txt".format(ts), "w")
    log1.write("args for data1: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path1,folder_path1,radius1,c1,epoches1))
    log1.write("args for data2: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path2,folder_path2,radius2,c2,epoches2))
    log2.write("args for data1: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path1,folder_path1,radius1,c1,epoches1))
    log2.write("args for data2: -cp1 {} -f1 {},-r1 {},-c1 {},-e {}\n".format(cell_path2,folder_path2,radius2,c2,epoches2))

    if (contin==False) or ((os.path.exists(folder_path1+'merge_cell_data.csv') and os.path.exists(folder_path1 + 'merge_cell_meta.csv')) == False):
        loss1 = merge_by_radius(cell_path1,folder_path1,radius1,method,meta_col)
        print("cell meta score for dataset1: {}\n".format(loss1))
        log1.write("cell metas score for dataset1: {}\n".format(loss1))
    else:
        print("dataset1 find files and skip merging")

    
    adata1 = pd.read_csv(folder_path1+"merge_cell_data.csv")
    cell_meta = pd.read_csv(folder_path1+"merge_cell_meta.csv")
    cell_meta = cell_meta.set_index(cell_meta.columns[0])
    adata1 = adata1.set_index(adata1.columns[0])
    adata1 = anndata.AnnData(adata1)
    adata1.obs['celltype'] = cell_meta.values.reshape(-1)
    
    
    if(contin==False) or ((os.path.exists(folder_path2+'merge_cell_data.csv') and os.path.exists(folder_path2 + 'merge_cell_meta.csv')) == False):
        loss2 = merge_by_radius(cell_path2,folder_path2,radius2,method,meta_col)
        print("cell meta score for dataset2: {}".format(loss2))
        log2.write("cell meta score for dataset2: {}\n".format(loss2))
    else:
        print("dataset2 find files and skip merging")

    adata2 = pd.read_csv(folder_path2+"merge_cell_data.csv")
    cell_meta = pd.read_csv(folder_path2+"merge_cell_meta.csv")
    cell_meta = cell_meta.set_index(cell_meta.columns[0])
    adata2 = adata2.set_index(adata2.columns[0])
    adata2 = anndata.AnnData(adata2)
    adata2.obs['celltype'] = cell_meta.values.reshape(-1)
    

    
    preprocessing_cluster(adata1,N_pcs=n_pca,resolution=resolution)
    preprocessing_cluster(adata2,N_pcs=n_pca,resolution=resolution)
    
    inter_gene = sort_data(adata1,adata2)

    tmp1 = calculate_cluster_centroid_for_genes(adata1,inter_gene,folder_path1)
    tmp2 = calculate_cluster_centroid_for_genes(adata2,inter_gene,folder_path2)
    
    ari = adjusted_rand_score(adata1.obs['celltype'].tolist(), adata1.obs['leiden'].tolist())
    print("ARI score for adata1: ", ari)
    log1.write("ARI score for adata1: "+ str(ari)+'\n')
    
    ari = adjusted_rand_score(adata2.obs['celltype'].tolist(), adata2.obs['leiden'].tolist())
    print("ARI score for adata2: ", ari)
    log2.write("ARI score for adata2: "+ str(ari)+'\n')

    meta_list1 = []
    clustername = adata1.obs['leiden'].unique().tolist()
    clustername = list(map(int, clustername))
    clustername.sort()
    for value in clustername:
        indices = [i for i, x in enumerate(adata1.obs['leiden']) if x == str(value)]
        t = [adata1.obs['celltype'].tolist()[index] for index in indices]
        most_common_element = max(t, key=t.count)
        meta_list1.append(most_common_element)
    np.save(folder_path1+'tree_merge.npy',meta_list1)
    
        
    meta_list2 = []
    clustername = adata2.obs['leiden'].unique().tolist()
    clustername = list(map(int, clustername))
    clustername.sort()
    for value in clustername:
        indices = [i for i, x in enumerate(adata2.obs['leiden']) if x == str(value)]
        t = [adata2.obs['celltype'].tolist()[index] for index in indices]
        most_common_element = max(t, key=t.count)
        meta_list2.append(most_common_element)
    np.save(folder_path2+'tree_merge.npy',meta_list2)
    
    
    v1 = pd.read_csv(folder_path1+"merge_labels.csv")
    meta = pd.read_csv(folder_path1+"merge_cell_meta.csv")
    meta = meta.set_index(meta.columns[0])
    meta
    lisan = []
    julei = []
    for i in range(len(v1)):
        lisan.append(meta.iloc[v1['label'][i]][0])
        julei.append(adata1.obs['leiden'].iloc[v1['label'][i]][0])
    v1['first']=lisan
    v1['second']=julei
    v1.to_csv(folder_path1+'meta_result.csv')
    
    v1 = pd.read_csv(folder_path2+"merge_labels.csv")
    meta = pd.read_csv(folder_path2+"merge_cell_meta.csv")
    meta = meta.set_index(meta.columns[0])
    meta
    lisan = []
    julei = []
    for i in range(len(v1)):
        lisan.append(meta.iloc[v1['label'][i]][0])
        julei.append(adata2.obs['leiden'].iloc[v1['label'][i]][0])
    v1['first']=lisan
    v1['second']=julei
    v1.to_csv(folder_path2+'meta_result.csv')
    
    if(contin==False) or ((os.path.exists(folder_path1 + 'dataxy.npy') and os.path.exists(folder_path1+'data1link.npy') and os.path.exists(folder_path1+'dataname.npy')) == False):
        get_Hyper_tree(folder_path1+'datas.data',1,tmp1.shape[1]+1,0,epoches1,save_path=folder_path1,c=0)
    else:
        print("dataset1 find files and skip embedding");

    if(contin==False) or ((os.path.exists(folder_path2 + 'dataxy.npy') and os.path.exists(folder_path2+'data1link.npy') and os.path.exists(folder_path1+'dataname.npy')) == False):
        get_Hyper_tree(folder_path2+'datas.data',1,tmp2.shape[1]+1,0,epoches2,save_path=folder_path2,c=0)
    else:
        print("dataset2 find files and skip embedding")

        
    nodes1,n1 = build_hyper_tree_from_folder(folder_path1)
    nodes2,n2 = build_hyper_tree_from_folder(folder_path2)

    merge_list1 = [];
    merge_list2 = [];
    nodes1[0] = search_tree(nodes1[0],c1,merge_list1)
    nodes2[0] = search_tree(nodes2[0],c2,merge_list2)
    
    for i in range(len(nodes1)):
        if(int(nodes1[i].name)<len(meta_list1)):
            nodes1[i].name= nodes1[i].name +'_'+ meta_list1[int(nodes1[i].name)];
            
    for i in range(len(nodes2)):
        if(int(nodes2[i].name)<len(meta_list2)):
            nodes2[i].name= nodes2[i].name +'_'+ meta_list2[int(nodes2[i].name)];  
    rate = 0;        
    if(alignment==1):
        rate = run_alignment(nodes1,nodes2,folder_path1,folder_path2,meta_list1,meta_list2);
    elif(alignment==2):
        rate = run_alignment_linear(nodes1,nodes2);
        
    log1.write("Alignment score: "+ str(rate)+'\n')
    log2.write("Alignment score: "+ str(rate)+'\n')
    log1.close()
    log2.close()