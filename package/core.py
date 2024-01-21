from read_data import *
from hyper import *
from alignment import *
from datasets.preprecossing import *
from scipy.spatial import KDTree
from sklearn.metrics import adjusted_rand_score
import calendar
import time

def alignment_process(cell_path1,cell_path2,folder_path1,folder_path2,radius1,radius2,c1,c2,epoches1,epoches2,contin=True,resolution=0.5,method='average',alignment=1):
    
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
        
        loss1 = merge_by_radius(cell_path1,folder_path1,radius1,method)
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
        
        loss2 = merge_by_radius(cell_path2,folder_path2,radius2,method)
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
    

    
    preprocessing_cluster(adata1,N_pcs=50,resolution=resolution)
    preprocessing_cluster(adata2,N_pcs=50,resolution=resolution)
    
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

        
    nodes1,n1 = build_hyper_tree(folder_path1)
    nodes2,n2 = build_hyper_tree(folder_path2)

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

    
    # T=tree_alignment(nodes1[0],nodes2[0],1);
    # minn = T.run_alignment();
    # T.show_ans();
    # ans = T.get_ans()
    # G=show_graph(ans,nodes1[0],nodes2[0]);
    # # G.show_fig()
    # G.save_fig(folder_path1+'alignment.png')
    # G.save_fig(folder_path2+'alignment.png')

    # log1.write('alignment anslist:{}\n'.format(ans))
    # log2.write('alignment anslist:{}\n'.format(ans))

    # log1.write("average cost for one node:{}\n".format(minn/(n1+n2)))
    # log2.write("average cost for one node:{}\n".format(minn/(n1+n2)))

    # print("average cost for one node:{}\n".format(minn/(n1+n2)))
    
    # c=0;z=0
    # for i,j in ans:
    #     i=int(i.split('_')[0])
    #     j=int(j.split('_')[0])
    #     if(i<len(meta_list1) and j <len(meta_list2)):
    #         c+=1
    #         if(meta_list1[i]==meta_list2[j]):
    #             z+=1;
    # print('correct alignment rate:{}'.format(z/c))
    

    log1.close()
    log2.close()