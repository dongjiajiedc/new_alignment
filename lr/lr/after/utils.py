from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import pandas as pd
import numpy as np
import random
import math
import Cytograph
from sklearn.neighbors import NearestNeighbors
# from StreamOperation import *
from splith5ad import*
from scipy.stats import binom
import scipy.sparse as sp


def st_mean(st_exp):
    value = st_exp.mean()
    # mask = st_exp>=value
    mask = st_exp>value
    st_exp[mask]=1
    st_exp[~mask]=0
    return st_exp

def st_median(st_exp):
    value = st_exp.median()
    mask = st_exp>=value
    st_exp[mask]=1
    st_exp[~mask]=0
    return st_exp

def preprocess_st(st_data,filtering):
    if filtering == "mean":
        st_data = st_data.apply(st_mean,axis=1)
    if filtering == "median":
        st_data = st_data.apply(st_median,axis=1)
    else:
        st_data[st_data>0]=1
    return st_data



def get_distance(st_meta,distance_threshold):
    if 'Z' in st_meta.columns:
        st_meta = st_meta.astype({'X':'float','Y':'float','Z':'float'})
        A = st_meta[["X","Y","Z"]].values
    if 'z' in st_meta.columns:
        st_meta = st_meta.astype({'x':'float','y':'float','z':'float'})
        A = st_meta[["x","y","z"]].values
    if 'X' in st_meta.columns:
        st_meta = st_meta.astype({'X':'float','Y':'float'})
        A = st_meta[["X","Y"]].values
    if 'x' in st_meta.columns:
        st_meta = st_meta.astype({'x':'float','y':'float'})
        A = st_meta[["x","y"]].values
    else:
        raise ValueError("the coordinate information must be included in st_meta")

    distA=squareform(pdist(A,metric='euclidean'))
    if distance_threshold:
        distance_rank = np.sort(distA, axis=1)
        dis_f = np.percentile(distance_rank[:,1:11].flatten(),95)
        distA = np.where(distA<=dis_f,distA,0)
    #distA[distA>distance_threshold] =float('-inf')
    if distA.sum()==0:
        raise ValueError("invalid distance threshold")
    dist_data = pd.DataFrame(data=distA, index = st_meta["cell"].values.tolist(), columns=st_meta["cell"].values.tolist())
    return dist_data

def co_exp(matrix):
    co_exp_ratio = np.count_nonzero(matrix, axis=1)/matrix.shape[1]
    return co_exp_ratio

def co_exp_list(exp_list):
    co_exp_ratio = np.count_nonzero(exp_list)/len(exp_list)
    return co_exp_ratio


def get_cell_list(st_meta):
    cell_type = list(set(st_meta["cell_type"].values.tolist()))
    #print(cell_type)
    sender_list = []
    receiver_list = []
    for i in range(len(cell_type)):
        receiver_list += [cell_type[i]]*((len(cell_type)))
        #sender_list += cell_type[0:[cell_type[i]]*(len(cell_type))]
        sender_list += cell_type[0:len(cell_type)]
    #print(sender_list)
    #print(receiver_list)
    return sender_list,receiver_list,cell_type

def get_cell_pair(st_meta, dist_data,cell_sender_name,cell_receiver_name,n_neighbor=10,min_pairs_ratio = 0.001):
    cell_sender = st_meta["cell"][st_meta["cell_type"] == cell_sender_name].values.tolist()
    cell_receiver = st_meta["cell"][st_meta["cell_type"]  == cell_receiver_name].values.tolist()
    pair_sender = []
    pair_receiver = []
    distance = []
    sender_type = []
    receiver_type = []

    for i in cell_sender:
        pairs = dist_data[i][dist_data[i]!=0].sort_values()[:n_neighbor].index.values.tolist()
        pair_sender += [i]*len(pairs)
        pair_receiver += pairs
        #print(dist_data[i][dist_data[i]!=0].sort_values()[:n_neighbor].values)
        distance += dist_data[i][dist_data[i]!=0].sort_values()[:n_neighbor].values.tolist()
        sender_type += [cell_sender_name]*len(pairs)
        receiver_type += [cell_receiver_name]*len(pairs)

    cell_pair = pd.DataFrame({'cell_sender':pair_sender, 'cell_receiver':pair_receiver, 'distance':distance, "sender_type":sender_type, "receiver_type":receiver_type})
    cell_pair = cell_pair[(cell_pair['cell_receiver'].isin(cell_receiver))]
    all_pair_number = len(cell_sender)*len(cell_receiver)
    pair_number = cell_pair.shape[0]
    flag = 1
    if pair_number <= all_pair_number * min_pairs_ratio:
        print(f"Cell pairs found between {cell_sender_name} and {cell_receiver_name} less than min_pairs_ratio!")
        flag = 0
    return cell_pair,flag

def find_sig_lr_in_chunk(file_path,lr_pair,cell_pair,gene_list, cell_name_list,per_num=1000,pvalue=0.05,chunk_size=10000,cells_per_file=10000):
    cell_ligand = cell_pair["cell_sender"].values.tolist()
    cell_receptor = cell_pair["cell_receiver"].values.tolist()

    n_chunks = len(cell_ligand) // chunk_size + (1 if len(cell_ligand) % chunk_size > 0 else 0)
    
    co_exp_value_sum = []
    # co_exp_number_sum = []
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(cell_ligand))
        current_ligand = cell_ligand[start_idx:end_idx]
        current_receptor = cell_receptor[start_idx:end_idx]

        data_ligand_sparse = get_gene_cell_matrix(current_ligand,lr_pair["ligand"].values.tolist(), cell_name_list, gene_list,file_path,cells_per_file)
        data_receptor_sparse = get_gene_cell_matrix(current_receptor,lr_pair["receptor"].values.tolist(), cell_name_list, gene_list,file_path,cells_per_file)

        # 元素对元素的乘法（Hadamard 乘法）
        lr_matrix_chunk = data_ligand_sparse.multiply(data_receptor_sparse).toarray()

        print(f"lr_matrix_chunk----------{lr_matrix_chunk.shape}")
        if lr_matrix_chunk.shape[0] > 1:
            co_exp_value = co_exp(lr_matrix_chunk)
            co_exp_value_sum.append(co_exp_value)
        else:
            co_exp_value = co_exp_list(lr_matrix_chunk[0])
            co_exp_value_sum.append(co_exp_value)
        print(f"co_exp_value_sum---------{len(co_exp_value_sum)}")

    
    final_co_exp_value=np.mean(co_exp_value_sum,axis=0)
    print(f"final_co_exp_value---------{final_co_exp_value.shape}")

    # co_exp_number = np.matrix((co_exp_value_sum))
    if len(lr_pair["ligand"].values.tolist()) > 1:
        final_co_exp_number = [x*len(cell_ligand) for x in final_co_exp_value]
    else:
        final_co_exp_number = final_co_exp_value*len(cell_ligand)
    

    gene_radio = pd.read_csv(f'{file_path}/gene_ratio.csv')

    
    per_ligand_ratio = gene_radio.loc[lr_pair["ligand"].values.tolist(),:]
    per_receptor_ratio = gene_radio.loc[lr_pair["receptor"].values.tolist(),:]

    co_exp_ratio = per_ligand_ratio.values*per_receptor_ratio.values
    print(f"co_exp_ratio=-------{ co_exp_ratio.shape}")

    n = cell_pair.shape[0]
    co_exp_value_binom = binom.ppf(0.95, n, co_exp_ratio)/n
    print(f"co_exp_value-------{co_exp_value_binom.shape}")
    co_exp_p = 1.0- binom.cdf(final_co_exp_number, n, co_exp_ratio.flatten())
    print(f"co_exp_p-----{co_exp_p.shape}")

        
    lr_pair = lr_pair.assign(co_exp_value=final_co_exp_value, co_exp_number=final_co_exp_number,co_exp_p=co_exp_p)
    lr_pair = lr_pair[lr_pair['co_exp_value'] > co_exp_value_binom.flatten()]

    return lr_pair

        
def get_gene_cell_matrix(request_cell_list, request_gene_list, cell_name_list, gene_list, file_path, cells_per_file=1000):
    """
    Retrieve a matrix of gene expression data for specified cells and genes.

    Args:
    request_cell_list (list of str): List of cell identifiers for which the gene expression data is to be retrieved.
    request_gene_list (list of str): List of gene identifiers for which the gene expression data is to be retrieved.
    cell_name_list (list of str): Complete list of cell names available in the dataset, used to map request_cell_list to indices.
    gene_list (list of str): Complete list of gene names available in the dataset, used to map request_gene_list to indices.
    file_path (str): Path to the file containing the sparse matrix data.
    cells_per_file (int, optional): Number of cells per split file, used to determine how the data is chunked. Default is 1000.

    Returns:
    scipy.sparse.csr_matrix: A sparse matrix containing the requested gene expression data for the specified cells.

    Description:
    This function is designed to extract a specific subset of gene expression data from a larger dataset.
    It utilizes helper functions to convert lists of cell and gene names into their corresponding indices within
    the full dataset. These indices are then used to retrieve the relevant section of the gene-cell matrix stored
    in a sparse format. This approach is efficient for handling large datasets commonly found in genomics and bioinformatics.
    """

    # Map the requested list of cell names to their corresponding indices in the full cell name list
    cell_index = get_cell_indices(request_cell_list, cell_name_list)
    
    # Map the requested list of gene names to their corresponding indices in the full gene list
    gene_index = get_gene_indices(request_gene_list, gene_list)
    # print(request_gene_list)
    
    # Retrieve the gene-cell matrix for the specified indices from the provided file path
    gene_cell_matrix = retrieve_gene_matrix(cell_index, gene_index, file_path, cells_per_file)
    
    # Return the sparse matrix containing the requested data
    return gene_cell_matrix


def get_gene_cell_matrix_2(request_cell_list, request_gene_list, cell_name_list, gene_list, file_path, cells_per_file=1000):
    """
    Retrieve a matrix of gene expression data for specified cells and genes.

    Args:
    request_cell_list (list of str): List of cell identifiers for which the gene expression data is to be retrieved.
    request_gene_list (list of str): List of gene identifiers for which the gene expression data is to be retrieved.
    cell_name_list (list of str): Complete list of cell names available in the dataset, used to map request_cell_list to indices.
    gene_list (list of str): Complete list of gene names available in the dataset, used to map request_gene_list to indices.
    file_path (str): Path to the file containing the sparse matrix data.
    cells_per_file (int, optional): Number of cells per split file, used to determine how the data is chunked. Default is 1000.

    Returns:
    scipy.sparse.csr_matrix: A sparse matrix containing the requested gene expression data for the specified cells.

    Description:
    This function is designed to extract a specific subset of gene expression data from a larger dataset.
    It utilizes helper functions to convert lists of cell and gene names into their corresponding indices within
    the full dataset. These indices are then used to retrieve the relevant section of the gene-cell matrix stored
    in a sparse format. This approach is efficient for handling large datasets commonly found in genomics and bioinformatics.
    """

    # Map the requested list of cell names to their corresponding indices in the full cell name list
    cell_index = get_cell_indices(request_cell_list, cell_name_list)
    # print(cell_index)
    # Map the requested list of gene names to their corresponding indices in the full gene list
    # gene_index = get_gene_indices(request_gene_list, gene_list)
    # print(request_gene_list)
    
    # Retrieve the gene-cell matrix for the specified indices from the provided file path
    gene_cell_matrix = retrieve_gene_matrix(cell_index, request_gene_list, file_path, cells_per_file)
    
    # Return the sparse matrix containing the requested data
    return gene_cell_matrix

    

def find_sig_lr(file_path,lr_pair,cell_pair,gene_list, cell_name_list,per_num=1000,pvalue=0.05,chunk_size=10000,cells_per_file=10000):

    #修改st_data
    cell_list = cell_pair["cell_sender"].values.tolist()

    if(len(cell_list)>chunk_size):
        return find_sig_lr_in_chunk(file_path,lr_pair,cell_pair,gene_list, cell_name_list,cells_per_file)

    #get ligand-sender matrix
    data_ligand_sparse = get_gene_cell_matrix(cell_list,lr_pair["ligand"].values.tolist(), cell_name_list, gene_list,file_path,cells_per_file)

    #get receptor-receiver matrix
    data_receptor_sparse = get_gene_cell_matrix(cell_pair["cell_receiver"].values.tolist(),lr_pair["receptor"].values.tolist(),cell_name_list, gene_list,file_path,cells_per_file)

    #转化成稀疏矩阵
    # data_ligand_sparse = sp.csr_matrix(data_ligand.values)
    # data_receptor_sparse = sp.csr_matrix(data_receptor.values)


    # 元素对元素的乘法（Hadamard 乘法）
    lr_matrix = data_ligand_sparse.multiply(data_receptor_sparse).toarray()


    if lr_matrix.shape[0] > 1:
        co_exp_value = co_exp(lr_matrix)
        
        co_exp_number = [x*len(lr_matrix[0]) for x in co_exp_value]
    else:
        co_exp_value = co_exp_list(lr_matrix[0])
        co_exp_number = co_exp_value*lr_matrix.shape[1]
    
    gene_radio = pd.read_csv(f'{file_path}/gene_ratio.csv',index_col=0)
    # print( gene_radio.head)

    
    per_ligand_ratio = gene_radio.loc[lr_pair["ligand"].values.tolist(),:]
    per_receptor_ratio = gene_radio.loc[lr_pair["receptor"].values.tolist(),:]

    co_exp_ratio = per_ligand_ratio.values*per_receptor_ratio.values
    print(f"co_exp_ratio=-------{ co_exp_ratio.shape}")

    n = cell_pair.shape[0]
    co_exp_value_binom = binom.ppf(0.95, n, co_exp_ratio)/n
    print(f"co_exp_value-------{co_exp_value_binom.shape}")

    co_exp_p = 1.0- binom.cdf(co_exp_number, n, co_exp_ratio.flatten())
    print(f"co_exp_p-----{co_exp_p.shape}")

        
    lr_pair = lr_pair.assign(co_exp_value=co_exp_value, co_exp_number=co_exp_number,co_exp_p=co_exp_p)
    lr_pair = lr_pair[lr_pair['co_exp_value'] > co_exp_value_binom.flatten()]

    return lr_pair

def get_cell_indices(receiver_list, cell_index_dict):
    """
    获取 receiver_list 中的元素在 cell_name_list 中的位置索引。

    Parameters:
    receiver_list (list): 需要查找位置的元素列表。
    cell_name_list (list): 包含元素的列表。

    Returns:
    list: receiver_list 中的元素在 cell_name_list 中的位置索引列表。
    """
    
    return [cell_index_dict[item] for item in receiver_list if item in cell_index_dict]


def get_gene_indices(receiver_list, gene_index_dict):
    """
    获取 receiver_list 中的元素在 cell_name_list 中的位置索引。

    Parameters:
    receiver_list (list): 需要查找位置的元素列表。
    gene_list (list): 包含元素的列表。

    Returns:
    list: receiver_list 中的元素在 gene_list 中的位置索引列表。
    """
    
    return [gene_index_dict[item] for item in receiver_list if item in gene_index_dict]


def find_high_exp_path_process_in_chunks(file_path, cell_list, cell_name_list, gene_list, pathway, chunk_size=4000,cells_per_file=10000):
    # 获取全部cell_index，但不一次性加载所有数据
    n_chunks = len(cell_list) // chunk_size + (1 if len(cell_list) % chunk_size > 0 else 0)

    # 初始化用于存储每个chunk结果的列表
    co_exp_results = []

    for i in range(n_chunks):
        # 计算当前块的开始和结束索引
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(cell_list))
        current_list = cell_list[start_idx:end_idx]

        # 对当前块加载和预处理数据
        # st_data_chunk = preprocess_stdata(file_path, current_indices, gene_list, cell_name_list)
        per_src_sparse  = get_gene_cell_matrix(current_list,pathway['src'].values.tolist(), cell_name_list, gene_list,file_path,cells_per_file)

        # per_src = retrieve_gene_matrix(current_indices,pathway['src'].values.tolist(),file_path,cells_per_file)

        per_dest_sparse  = get_gene_cell_matrix(current_list,pathway['dest'].values.tolist(), cell_name_list, gene_list,file_path,cells_per_file)


        # 元素对元素的乘法（Hadamard 乘法）
        per_matrix = per_src_sparse.multiply(per_dest_sparse).toarray()
        print(f"per_matrix-------{per_matrix.shape}")
        

        if per_matrix.shape[0]>1:
            per_exp = co_exp(per_matrix)
            print(f"per_exp-----{per_exp.shape}")
        else:
            per_exp = co_exp_list(per_matrix[0])
            print(f"per_exp-----{per_exp}")

        # 将当前块的结果添加到结果列表中
        co_exp_results.append(per_exp)

    # 累加合并所有块的结果
    final_co_exp_ratio = np.mean(co_exp_results, axis=0)
    print(f"final_co_exp_ratio-------{final_co_exp_ratio.shape}")
    pathway["co_exp_ratio"] = final_co_exp_ratio

    return pathway


def find_high_exp_path(file_path,pathway,per_src_index,per_dest_index,receiver_list,gene_list, cell_name_list,chunk_size=4000,cells_per_file=10000):
    ## 修改 st_data
    cell_list_index = get_cell_indices(receiver_list,cell_name_list)


    # n = len(pathway['src'].values.tolist())

    # per_exp = np.zeros((n, 1), dtype=int)

    # for cell_id in cell_list_index:
    #     gene_vector = retrieve_cell_point(cell_id,file_path,cells_per_file).toarray()

    #     per_src = gene_vector[per_src_index,:]!= 0
    #     per_dest = gene_vector[per_dest_index,:]!= 0
    #     result = np.logical_and(per_src, per_dest)

    #     per_exp = per_exp+result

    # per_exp = per_exp / len(cell_list_index)


    per_src_sparse  = get_gene_cell_matrix_2(receiver_list,per_src_index, cell_name_list, gene_list,file_path,cells_per_file)

    per_dest_sparse = get_gene_cell_matrix_2(receiver_list,per_dest_index, cell_name_list, gene_list,file_path,cells_per_file)

    # 元素对元素的乘法（Hadamard 乘法）
    per_matrix = per_src_sparse.multiply(per_dest_sparse).toarray()
    # print(f"per_matrix-------{per_matrix.shape}")

    per_exp = co_exp(per_matrix)
    print(f"per_exp-------{per_exp.shape}")
    pathway["co_exp_ratio"] = per_exp
    return pathway

def get_score(sig_lr_pair,tf):
    tf_score = tf.groupby(by=['receptor'])['score_rt'].sum()
    tf_score = tf_score*(-1)
    sig_lr_pair["lr_score"] = 1-sig_lr_pair["co_exp_p"]
    rt_score = tf_score.map(lambda x:  1/(1 + math.exp(x)))
    rt = pd.DataFrame({'receptor':rt_score.index, 'rt_score':rt_score.values})  
    result = pd.merge(sig_lr_pair, rt, on=['receptor'])
    result["score"]= result.apply(lambda x: math.sqrt(x.lr_score*x.rt_score), axis=1)
    return result


def post_process(results):
    
    df_empty = pd.DataFrame(columns=["ligand","receptor","species","cell_sender","cell_receiver","co_exp_value", "co_exp_number", "co_exp_p",  "lr_score", "rt_score", "score"])
    pair_empty = pd.DataFrame(columns=["cell_sender","cell_receiver","distance","sender_type","receiver_type"])
    obj = {}
    for result in results:
        if result is not None:
            df_empty = pd.concat([df_empty, result[0]], axis=0)
            pair_empty = pd.concat([pair_empty, result[1]], axis=0)
    obj['lr_score'] = df_empty
    pair_empty.rename(columns={"cell_sender": "cell_sender_id", "cell_receiver": "cell_receiver_id"},inplace=True)
    obj['pair_distance'] = pair_empty

    return obj



def process_sender_receiver(i, lr_pair, distances_knn, indices_knn,sender_list, receiver_list, st_meta_origin, pathway, valid_st_data, st_data, max_hop):
    cell_sender = sender_list[i]
    cell_receiver = receiver_list[i]


    ####
    cell_pair, flag  = get_cell_pair_knn(st_meta_origin,cell_sender,cell_receiver,distances_knn,indices_knn)
            
    if flag == 0:
        return
    ####
    
    print (f"The cell pair number found between {cell_sender} and {cell_receiver} is {cell_pair.shape[0]}")
    f_path = find_high_exp_path(pathway, cell_pair["cell_receiver"].values.tolist(), valid_st_data)
    f_path = f_path[f_path['co_exp_ratio']>0.10]
    path_gene = list(set(f_path['src'].values).union(set(f_path['dest'].values)))
    pathway_graph = Cytograph.PathGraph(path_gene,max_hop)
    pathway_graph.built_edge(f_path)
    receptor_gene = pathway_graph.find_valid_lr_pair()
    #de_bug = lr_pair[lr_pair['receptor'].isin(f_path["src"])]
    lr_pair = lr_pair[lr_pair['receptor'].isin(receptor_gene)]


    if lr_pair.shape[0] == 0:
        print (f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} because of no downstream transcriptional factors found for receptors!")
        return
    else:
        print (f"the number of valid pathway number between {cell_sender} and {cell_receiver} is: {lr_pair.shape[0]}")
        #find cell pair
        lr_pair.insert(0, 'cell_sender', cell_sender)
        lr_pair.insert(1, 'cell_receiver', cell_receiver)



        sig_lr_pair = find_sig_lr(st_data,lr_pair,cell_pair)

        if sig_lr_pair.shape[0] == 0:
            print (f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression")
            return
        print (f"The ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression is {sig_lr_pair.shape[0]}")

        tf = pathway_graph.find_lr_tf(sig_lr_pair)

        path_score = get_score(sig_lr_pair,tf)

        print(f"{cell_receiver} and {cell_sender} done")
    return path_score, cell_pair


def get_knn_result(st_meta,k=11):
    '''
    **input: 
        st_meta: index由细胞名称组成,x,y列代表坐标，cell_type, cell列同样是细胞名字
        k: 邻居数+1(排除自身)

    ** return
        distances_knn: 每个点和邻居的距离
        indices_knn: 每个店的邻居的index!!!这里的index是数字,并不是前面的名字(和st_meta中的index不同)
    '''
    # 获得x,y坐标 array
    id_x_y_array = st_meta[['x', 'y']].values

    # 设置k值为11
    k = 11

    # 初始化k最近邻模型
    knn_model = NearestNeighbors(n_neighbors=k,metric='euclidean')

    # 训练k最近邻模型
    knn_model.fit(id_x_y_array)

    # 使用模型查找每个数据点的前5个邻居
    distances_knn, indices_knn = knn_model.kneighbors(id_x_y_array)
    # 排除自身
    indices_knn = indices_knn[:, 1:]
    distances_knn = distances_knn[:, 1:]
    print("knn done")

    return distances_knn,indices_knn


def get_cell_pair_knn(st_meta_origin,cell_sender,cell_receiver,distances_knn,indices_knn,min_pairs_ratio = 0.001):

    ####
    sender_array = np.array(st_meta_origin[st_meta_origin['cell_type'] == cell_sender].index)

    receiver_array = np.array(indices_knn[sender_array,:])

    all_pair_number = min(sender_array.shape[0]*receiver_array.shape[0],st_meta_origin.shape[0])
    print(f"all_pair_number ---- {all_pair_number}" )
    # 初始化一个空列表，用于存储每行的结果
    result_indices = []

    # 遍历receiver_array的每一行
    for i, row in enumerate(receiver_array):
        # 从当前行中获取id值
        ids = row.tolist()
        sender = sender_array[i]
        
        # 查找对应id下cell_type为'pDCs'的行索引
        # 使用ids直接筛选出满足条件的行，并获取它们的索引
        # indices_receiver = st_meta_1.loc[ids][st_meta_1.loc[ids]['cell_type'] == cell_receiver].index.tolist()
        indices_in_ids = np.where((st_meta_origin.loc[ids]['cell_type'] == cell_receiver).values)[0]

        combined_indices = [[sender, ids[index],distances_knn[sender][index]] for index in indices_in_ids]
        
        # 将结果添加到结果列表中
        result_indices.extend(combined_indices)

    result_array=np.array(result_indices)
    flag=1
    if result_array.shape[0]<=all_pair_number*min_pairs_ratio:
        print(f"Cell pairs found between {cell_sender} and {cell_receiver} less than min_pairs_ratio!")
        flag=0
        return [],flag


    # 将索引数组拆分为发送者索引和接收者索引
    sender_indices = result_array[:, 0]
    receiver_indices = result_array[:, 1]
    distances_indices = result_array[:, 2]

    # 通过索引直接获取对应的值，构建DataFrame
    cell_pair = pd.DataFrame({
        'cell_sender': st_meta_origin.loc[sender_indices, 'cell'].values,
        'cell_receiver': st_meta_origin.loc[receiver_indices, 'cell'].values,
        'sender_type': st_meta_origin.loc[sender_indices, 'cell_type'].values,
        'receiver_type': st_meta_origin.loc[receiver_indices, 'cell_type'].values,
        'distance': distances_indices,
        'sender_id': sender_indices,
        'receiver_id': receiver_indices
    })
    return cell_pair,flag
