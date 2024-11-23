import scanpy as sc
import pandas as pd
import numpy as np
import os
import scipy
import random
import gc
import argparse

def preprocessh5ad(h5ad_filePath,output_dir,pathway_path,species='Human'):
    adata = sc.read_h5ad(h5ad_filePath)
    st_data = adata.to_df()
    st_data = st_data.transpose()

    st_data = st_data[st_data.apply(np.sum,axis=1)!=0]
    st_gene = st_data.index.values.tolist()
    # print(f"st_gene------{len(st_gene)}---{st_gene[0]}")

    pathway = pd.read_table(pathway_path,delimiter='\t',encoding= 'unicode_escape')
    pathway = pathway[["src","dest","src_tf","dest_tf"]][pathway['species'] == species].drop_duplicates()
    pathway = pathway[(pathway['src'].isin(st_gene))&(pathway['dest'].isin(st_gene))]


    valid_gene = list(set(pathway['src'].values).union(set(pathway['dest'].values)))

    # print(f"valid_gene------{len(valid_gene)}---{valid_gene[0]}")


    adata = adata[:, valid_gene]
    print(adata)
    # print(f"adata------{adata.shape}---")


    if isinstance(adata.X, scipy.sparse.spmatrix):
        col_means = np.mean(adata.X.toarray(), axis=0)
    else:
        col_means = np.mean(adata.X, axis=0)

    # 步骤3: 二值化数据
    # 再次检查 adata.X 是否稀疏，因为处理方式略有不同
    if isinstance(adata.X, scipy.sparse.spmatrix):
        # 转换为密集矩阵进行操作
        adata_dense = adata.X.toarray()
        binary_data = (adata_dense >= col_means).astype(int)
        gene_ratio = np.count_nonzero(binary_data, axis=0) / binary_data.shape[0]
        
        # 将处理后的数据转换回原始稀疏格式
        adata.X = scipy.sparse.csr_matrix(binary_data)
        # adata_copy = adata.copy()
        # adata_copy.X = scipy.sparse.csr_matrix(binary_data)

    else:
        adata.X = (adata.X > col_means).astype(int)
        # 计算每个基因的非零比率
        gene_ratio = np.count_nonzero(adata.X, axis=0) / adata.X.shape[0]

    gene_ratio_df = pd.DataFrame(gene_ratio , index=valid_gene, columns=['Non-Zero Ratio'])

    #检查目录是否存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存到CSV文件，不包含行索引
    gene_ratio_df.to_csv(f"{output_dir}/gene_ratio.csv", index=True, index_label="Gene Name")

    return adata



def split_h5ad(input_file, output_dir, pathway_path, cells_per_file=10000, species='Human'):
    # 加载h5ad文件
    adata = preprocessh5ad(input_file, output_dir, pathway_path, species)
    
    # 计算需要分割成多少个文件
    n_cells = adata.shape[0]
    n_files = np.ceil(n_cells / cells_per_file).astype(int)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存元数据为CSV文件
    if 'meta' in adata.obsm_keys():
        metadata_path = os.path.join(output_dir, 'st_meta.csv')
        adata.obsm['meta'].to_csv(metadata_path)
    else:
        print("can not find meta in obsm")
    
    # 保存细胞名称
    cell_name_path = os.path.join(output_dir, 'cell_name.csv')
    pd.DataFrame(adata.to_df().index.values.tolist()).to_csv(cell_name_path, header=False)
    # pd.DataFrame(adata.obs_names).to_csv(cell_name_path, header=False)
    
    # 保存基因名称
    gene_list_path = os.path.join(output_dir, 'gene_list.csv')
    # pd.DataFrame(adata.var.index.tolist()).to_csv(gene_list_path, header=False)
    pd.DataFrame(adata.to_df().columns.values.tolist()).to_csv(gene_list_path, header=False)

    
    # 分割文件并保存
    for i in range(n_files):
        start_idx = i * cells_per_file
        end_idx = min((i + 1) * cells_per_file, n_cells)
        adata_sub = adata[start_idx:end_idx]
        
        # 将数据转换为CSC格式的稀疏矩阵，然后保存
        # 转换为CSC格式的稀疏矩阵
        if not scipy.sparse.issparse(adata_sub.X):
            matrix_sparse = scipy.sparse.csc_matrix(adata_sub.X)
        else:
            matrix_sparse = adata_sub.X.tocsc()
        
        # 保存稀疏矩阵为.npz文件
        matrix_file_path = os.path.join(output_dir, f'split_{i}.npz')
        scipy.sparse.save_npz(matrix_file_path, matrix_sparse)


def split_csv(input_file, output_dir, pathway_path, cells_per_file=10000,species='Human'):
    # 加载CSV文件
    st_data = pd.read_csv(input_file,index_col=0)

    st_data = st_data[st_data.sum(axis=1) != 0]

    st_gene = st_data.index.values.tolist()
    # print(f"st_gene------{len(st_gene)}---{st_gene[0]}")

    pathway = pd.read_table(pathway_path,delimiter='\t',encoding= 'unicode_escape')
    pathway = pathway[["src","dest","src_tf","dest_tf"]][pathway['species'] == species].drop_duplicates()
    pathway = pathway[(pathway['src'].isin(st_gene))&(pathway['dest'].isin(st_gene))]


    valid_gene = list(set(pathway['src'].values).union(set(pathway['dest'].values)))

    # print(f"valid_gene------{len(valid_gene)}---{valid_gene[0]}")
    # print(f"st_data------{st_data.shape}---")
    

    st_data = st_data.loc[valid_gene]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 保存基因名称
    gene_list_path = os.path.join(output_dir, 'gene_list.csv')
    pd.DataFrame(st_data.index.values.tolist()).to_csv(gene_list_path, header=False)

    st_data = st_data.T

    print(f"st_data------{st_data.shape}---")
    
    # 计算需要分割成多少个文件
    n_cells = st_data.shape[0]  # 行数代表单元数量
    n_files = np.ceil(n_cells / cells_per_file).astype(int)
    
    
    # 分割文件并保存为稀疏矩阵（NPZ）
    for i in range(n_files):
        start_idx = i * cells_per_file
        end_idx = min((i + 1) * cells_per_file, n_cells)
        df_sub = st_data.iloc[start_idx:end_idx]
        
        # 将DataFrame转换为稀疏矩阵
        matrix = scipy.sparse.csr_matrix(df_sub.values)
        
        # 保存稀疏矩阵为.npz文件
        file_path = os.path.join(output_dir, f'split_{i}.npz')
        scipy.sparse.save_npz(file_path, matrix)


def retrieve_cell_matrix(cell_list, split_files_dir, cells_per_file=10000):
    # 创建一个字典来按文件标号组织细胞ID
    cells_by_file = {}
    for cell_id in cell_list:
        # 计算cell_id所在的文件标号
        file_index = int(cell_id.split('_')[-1]) // cells_per_file
        
        # 计算在文件中的序号（如果需要）
        cell_index = int(cell_id.split('_')[-1]) % cells_per_file
        
        # 将cell_id添加到对应文件标号的列表中
        if file_index not in cells_by_file:
            cells_by_file[file_index] = []
            cells_by_file[file_index].append(cell_index)
        else:
            cells_by_file[file_index].append(cell_index)

    dfs = []
    
    # 遍历每个文件标号，读取并处理相应的细胞数据
    for file_index in cells_by_file:
        file_path = os.path.join(split_files_dir, f'split_{file_index}.npz')
        # 读取对应的分割文件
        adata = sc.read_h5ad(file_path)
        
        selected_adata = adata[cells_by_file[file_index]]
        
        # 对selected_adata进行所需的处理
        # 例如，可以打印出来，或者将其保存到文件中
        dfs.append(selected_adata.to_df())
    
    combined_df = pd.concat(dfs)
    print( combined_df)
    return combined_df



def retrieve_cell_matrix_2(cell_list, split_files_dir, cells_per_file=10000):
    # 创建一个字典来按文件标号组织细胞ID
    cells_by_file = {}
    for cell_id in cell_list:
        # 计算cell_id所在的文件标号
        file_index = int(cell_id.split('_')[-1]) // cells_per_file
        
        # 计算在文件中的序号（如果需要）
        cell_index = int(cell_id.split('_')[-1]) % cells_per_file
        
        # 将cell_id添加到对应文件标号的列表中
        if file_index not in cells_by_file:
            cells_by_file[file_index] = [cell_index]
        else:
            cells_by_file[file_index].append(cell_index)

    # 初始化一个空的DataFrame列表
    dfs = []
    
    # 遍历每个文件标号，读取并处理相应的细胞数据
    for file_index in cells_by_file:
        file_path = os.path.join(split_files_dir, f'split_{file_index}.npz')
        # 加载对应的分割稀疏矩阵文件
        matrix_sparse = scipy.sparse.load_npz(file_path)
        
        # 根据cells_by_file中的索引选择数据
        selected_matrix = matrix_sparse[cells_by_file[file_index], :]
        
        # 将选中的稀疏矩阵转换为DataFrame
        selected_df = pd.DataFrame(selected_matrix.toarray())
        
        # 将DataFrame添加到列表中
        dfs.append(selected_df)

        # 清理内存
        del matrix_sparse, selected_matrix, selected_df
        gc.collect()
    
    # 将所有DataFrame合并为一个DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # 打印并返回合并后的DataFrame
    print(combined_df)
    return combined_df



def retrieve_gene_matrix(cell_list, gene_indices, split_files_dir, cells_per_file=10000):
    # 创建一个字典来按文件标号组织细胞ID
    cells_by_file = {}
    for cell_id in cell_list:
        # 计算cell_id所在的文件标号
        file_index = cell_id // cells_per_file
        
        # 计算在文件中的序号（如果需要）
        cell_index = cell_id % cells_per_file
        
        # 将cell_id添加到对应文件标号的列表中
        if file_index not in cells_by_file:
            cells_by_file[file_index] = []
            cells_by_file[file_index].append(cell_index)
        else:
            cells_by_file[file_index].append(cell_index)

    dfs = []
    
    # 遍历每个文件标号，读取并处理相应的细胞数据
    for file_index in cells_by_file:
        file_path = os.path.join(split_files_dir, f'split_{file_index}.npz')
        # 加载对应的分割稀疏矩阵文件
        matrix_sparse = scipy.sparse.load_npz(file_path)
        
        # 根据cells_by_file中的索引选择数据
        selected_matrix = matrix_sparse[cells_by_file[file_index], :]
        
        # print(selected_adata)
        
        selected_gene = selected_matrix[:, gene_indices]


        # 对selected_adata进行所需的处理
        # 例如，可以打印出来，或者将其保存到文件中
        dfs.append(selected_gene)

        # 清理内存
        del matrix_sparse, selected_matrix, selected_gene
        gc.collect()
    
    # print(f"dfs------{dfs}")
    combined_sparse_matrix = scipy.sparse.vstack(dfs).transpose()
    # print(f"combined_sparse_matrix---------{combined_sparse_matrix.toarray().shape}")

    # print(combined_sparse_matrix)
    return combined_sparse_matrix

def retrieve_cell_point(cell_id, split_files_dir, cells_per_file=10000):
        
    file_index = cell_id // cells_per_file
    
    # 计算在文件中的序号（如果需要）
    cell_index = cell_id % cells_per_file
        
    
    file_path = os.path.join(split_files_dir, f'split_{file_index}.npz')
        # 加载对应的分割稀疏矩阵文件
    matrix_sparse = scipy.sparse.load_npz(file_path)
        
        # 根据cells_by_file中的索引选择数据
    selected_cell_vector = matrix_sparse[cell_index, :]
        
    # print(f"dfs------{dfs}")
    # combined_sparse_matrix = scipy.sparse.vstack(dfs).transpose()
    # print(f"combined_sparse_matrix---------{combined_sparse_matrix.toarray().shape}")

    # print(combined_sparse_matrix)
    return selected_cell_vector.T




def main():
    print("start spilting data")
    parser = argparse.ArgumentParser(description="Split h5ad file into multiple smaller files.")
    parser.add_argument('input_file', type=str, help='Path to the input .h5ad file')
    parser.add_argument('output_dir', type=str, help='Directory where the output files will be saved')
    parser.add_argument('pathways_file', type=str, help='Path to the pathways .tsv file')
    parser.add_argument('cells_per_file', type=int, help='Number of cells per split file')
    parser.add_argument('species', type=str, nargs='?', default='Human',help='Choose species Mouse or Human [default: Human]')

    
    args = parser.parse_args()

    # 获取文件扩展名
    _, file_ext = os.path.splitext(args.input_file)

    # 根据文件类型调用不同的函数
    if file_ext == '.h5ad':
        if not args.pathways_file:
            print("Pathways file is required for h5ad files.")
            return
        split_h5ad(args.input_file, args.output_dir, args.pathways_file, args.cells_per_file,args.species)
    elif file_ext == '.csv':
        if not args.pathways_file:
            print("Pathways file is required for h5ad files.")
            return
        split_csv(args.input_file, args.output_dir, args.pathways_file, args.cells_per_file,args.species)
    else:
        print(f"Unsupported file type: {file_ext}")



if __name__ == '__main__':
    main()


# python3 splith5ad.py demo/pdac.h5ad demo/pdac_split demo/pathways.tsv 1000
# python3 /home/jianganna/cytotour_new_knn/CytoTour.py /home/jianganna/cytotour_new_knn/lr_database.csv /home/jianganna/cytotour_new_knn/pathways.tsv /data1/jianganna/3.h5ad -s Mouse --o "/home/jianganna/cytotour_new_knn/output_1"
# python3 /home/jianganna/cytotour_new_knn/splith5ad.py /data1/jianganna/3.h5ad /data1/jianganna/pdac_split /home/jianganna/cytotour_new_knn/pathways.tsv 10000 Mouse

# split_h5ad('pdac.h5ad', 'pdac_split','pathways.tsv',cells_per_file=1000)
# split_h5ad('pdac.h5ad', 'pdac_split','pathways.tsv',cells_per_file=1000)

# # 定义范围和数量
# total_cells = 4000
# number_to_select = 100  # 选择的数量，可以调整

# # 生成1到100的数字列表
# numbers = list(range(1, total_cells + 1))

# # 随机选择100个不重复的数字
# selected_numbers = random.sample(numbers, number_to_select)

# # 将选中的数字转换为字符串格式，添加到cell_list
# cell_list = [f'{num}' for num in selected_numbers]

# retrieve_cell_matrix_2(cell_list,'pdac_split',cells_per_file=1000)
# retrieve_gene_matrix(['1','2','4000'],['PAN3','ABCB9'],'pdac_split',cells_per_file=1000)

# retrieve_gene_ratio(['1','2','4000'],['PAN3','ABCB9'],'pdac_split',cells_per_file=1000)
