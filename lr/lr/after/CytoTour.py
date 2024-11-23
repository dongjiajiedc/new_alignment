"""CytoTour
Usage:
    CytoTour.py <lr_pair> <pathwaydb> <spilt_dir> <cells_per_file> <st_file> ... [--species=<sn>]  [--filtering=<fn>]  [--distance_threshold]  [--parallel]  [--max_hop=<mn>]  [--cell_sender=<cn>]  [--cell_receiver=<rn>]  [--out=<fn>]  [--core=<cn>]
    CytoTour.py (-h | --help)
    CytoTour.py --version

Options:
    -s --species=<sn>   Choose species Mouse or Human [default: Human].
    -f --filtering=<fn>   The thershold of valid expression count, choose median, mean or null [default: mean].
    -d --distance_threshold   Using 95% largest distance to filter cell neighbors.
    -p --parallel   Using parallel.
    -m --max_hop=<mn>   Set the max hop to find lr pairs with tf [default: 3].
    -c --cell_sender=<cn>   The cell type of sender [default: all].
    -r --cell_receiver=<rn>   The cell type of receiver [default: all].
    -o --out=<fn>   Outdir [default: .].
    -n --core=<cn>   Core number [default: 4].
    -h --help   Show this screen.
    -v --version    Show version.
"""

import pandas as pd
import numpy as np
from docopt import docopt
from utils import *
import Cytograph
import multiprocessing as mp
import datetime
import anndata as ad
import pickle
from sklearn.neighbors import NearestNeighbors
# from StreamOperation import *
from splith5ad import*


def main(arguments):
    print("start filtering LRIs with spatial data")
    st_files = arguments.get("<st_file>")
    spilt_dir = arguments.get("<spilt_dir>")
    cells_per_file = int(arguments.get("<cells_per_file>"))
    lr_pair = arguments.get("<lr_pair>")
    pathwaydb = arguments.get("<pathwaydb>")
    cell_sender = str(arguments.get("--cell_sender"))
    cell_receiver = str(arguments.get("--cell_receiver"))
    parallel = arguments.get("--parallel")
    filtering = str(arguments.get("--filtering"))
    species = str(arguments.get("--species"))
    distance_threshold = arguments.get("--distance_threshold")
    out_dir = str(arguments.get("--out"))
    max_hop = int(arguments.get("--max_hop"))
    n_core = int(arguments.get("--core"))
    starttime = datetime.datetime.now()


    if  max_hop is None:
        if (species == "Mouse"):
            max_hop = 4
        else:
            max_hop = 3
    print("reading data")
    if (len(st_files)==1):
        if st_files[0].endswith("h5ad"):
            # adata = ad.read_h5ad(st_files[0])
            # st_data = adata.to_df()
            # st_data = st_data.transpose(

            # st_meta = adata.obsm['meta']
            # st_meta['cell'] = st_meta.index.values.tolist()
            #st_meta = st_meta[st_meta["cell_type"].isin(["Acinar_cells","pDCs"])]

            # split_h5ad(st_files[0], spilt_dir,pathwaydb,cells_per_file=1000, species=species)

            # st_meta_dict = pd.read_csv(f'{spilt_dir}/st_meta.csv',index_col=0)
            

            
            # print(st_meta_dict.head)

            st_meta_origin = pd.read_csv(f'{spilt_dir}/st_meta.csv')
            st_meta_origin.columns = st_meta_origin.columns.str.lower()

            if 'Unnamed: 0' in st_meta_origin.columns:
                st_meta_origin.rename(columns={'Unnamed: 0': 'cell'}, inplace=True)

            print(st_meta_origin.head)

            if '_index' in st_meta_origin.columns:
                # 替换列名为 "cell"
                st_meta_origin = st_meta_origin.rename(columns={'_index': 'cell'})
                print("替换后的 DataFrame:")
                print(st_meta_origin)
            else:
                print("DataFrame 中不存在名为 '_index' 的列。")

            st_meta = st_meta_origin.set_index('cell', inplace=False)
            print(st_meta_origin.columns)

            #---
            #获得基因名称列表
            st_gene = pd.read_csv(f"{spilt_dir}/gene_list.csv",header=None,index_col=0).squeeze().tolist()
            print(f"st_gene_len----{len(st_gene)}")
            ## 获得细胞名称列表 
            cell_name_list = st_meta_origin['cell'].values.tolist()
            print(f"cell_name_list----{len(cell_name_list)}")

            if 'cell_type' not in st_meta.columns:
                TypeError("There is no column named 'cell_type' in st_meta file")
        else:
            TypeError("st_file should be st_meta.csv and st_data.csv or xxx.h5ad")
    elif len(st_files)==2:
        if st_files[0].endswith(".csv"):
            st_meta = pd.read_csv(st_files[0])
            if 'cell_type' not in st_meta.columns:
                TypeError("There is no column named 'cell_type'")
        else:
            TypeError("st_file should be st_meta.csv and st_data.csv or xxx.h5ad")

        # split_csv(st_files[1], spilt_dir,pathwaydb,cells_per_file=cells_per_file, species=species)
        st_gene = pd.read_csv(f"{spilt_dir}/gene_list.csv",header=None,index_col=0).squeeze().tolist()
        cell_name_list = st_meta['cell'].values.tolist()
        # st_data = pd.read_csv(st_files[1],index_col=0)

        ##process data
        # st_data = st_data[st_data.apply(np.sum,axis=1)!=0]
        # st_gene = st_data.index.values.tolist()


    print("reading data done")

    ##read data
    #st_meta = st_meta[st_meta["label"] != "less nFeatures"]

    ##------------------
    # st_data = st_data[st_data.apply(np.sum,axis=1)!=0]
    # st_gene = st_data.index.values.tolist()

    ## -------------
    lr_pair = pd.read_csv(lr_pair)
    lr_pair = lr_pair[lr_pair['species'] == species]
    pathway = pd.read_table(pathwaydb,delimiter='\t',encoding= 'unicode_escape')


    ##filtering
    pathway = pathway[["src","dest","src_tf","dest_tf"]][pathway['species'] == species].drop_duplicates()
    # print(f"pathway------{pathway.shape}")
    # print(f"st_gene_len----{st_gene[:5]}")

    pathway = pathway[(pathway['src'].isin(st_gene))&(pathway['dest'].isin(st_gene))]
    # print(f"pathway------{pathway.shape}")


    # valid_gene = list(set(pathway['src'].values).union(set(pathway['dest'].values)))
    # st_data = preprocess_st(st_data,filtering)
    # valid_st_data = st_data.loc[valid_gene,:]
    lr_pair = lr_pair[lr_pair['receptor'].isin(st_gene)&lr_pair['ligand'].isin(st_gene)]
    ##get cell list
    if cell_sender=="all" and cell_receiver=="all":
        sender_list,receiver_list,cell_type = get_cell_list(st_meta)
        print(f"The unique celltype list is {cell_type}")
    else:
        #print(f"The number of unique celltypes is {len(cell_type)}")
        sender_list=[cell_sender]
        receiver_list=[cell_receiver]

    #####
    # dist_data = get_distance(st_meta,distance_threshold)
        
    #####
    lr_pair_all = lr_pair



    if not parallel:
        # report_columns = ["receptor", "tf"] + st_meta["cell"].values.tolist()
        # rtf_report = pd.DataFrame(columns=report_columns)
        # all_lr_score = pd.DataFrame(columns=["ligand","receptor","species","cell_sender","cell_receiver","co_exp_value", "co_exp_number", "co_exp_p",  "lr_score", "rt_score", "score"])
        
        # obj = {'cell_pair': {} }

        # ###
        # ###
        distances_knn,indices_knn = get_knn_result(st_meta,k=11)

        gene_index_dict = {gene: idx for idx, gene in enumerate(st_gene)}
        cell_index_dict = {cell_name: idx for idx, cell_name in enumerate(cell_name_list)}


        per_src_index =  [gene_index_dict[item] for item in pathway['src'].values.tolist() if item in gene_index_dict]
        per_dest_index =  [gene_index_dict[item] for item in pathway['dest'].values.tolist() if item in gene_index_dict]




        for i in range(len(sender_list)):
            all_lr_score = pd.DataFrame(columns=["ligand","receptor","species","cell_sender","cell_receiver","co_exp_value", "co_exp_number", "co_exp_p",  "lr_score", "rt_score", "score"])
        
            obj = {'cell_pair': {} }
            cell_sender = sender_list[i]
            cell_receiver = receiver_list[i]

            cell_pair, flag  = get_cell_pair_knn(st_meta_origin,cell_sender,cell_receiver,distances_knn,indices_knn)
            
            if flag == 0:
                continue
            ####

            # for each cell type receiver, find all edge with expression ration>0.1
            # cell_pair,flag = get_cell_pair(st_meta,dist_data,cell_sender,cell_receiver)
            
            print (f"The cell pair number found between {cell_sender} and {cell_receiver} is {cell_pair.shape[0]}")

            ### 修改了st_data
            # print(f"pathway------{pathway.shape}")
            find_ligand_recepotor_pairs_time_start = datetime.datetime.now()
            # f_path = find_high_exp_path(st_files[0] ,pathway, cell_pair["cell_receiver"].values.tolist(), st_gene, cell_name_list= cell_name_list)
            f_path = find_high_exp_path(spilt_dir,pathway,per_src_index,per_dest_index,cell_pair["cell_receiver"].values.tolist(), gene_index_dict, cell_index_dict ,cells_per_file=cells_per_file)

            find_ligand_recepotor_pairs_time = datetime.datetime.now()
            print(f"find ligand-recepotor pairs time-------------{(find_ligand_recepotor_pairs_time-find_ligand_recepotor_pairs_time_start).seconds}")

            
            f_path = f_path[f_path['co_exp_ratio']>0.1]

            # print(f"f_path-----{f_path.shape}")
            path_gene = list(set(f_path['src'].values).union(set(f_path['dest'].values)))
            # print(f"path_gene--------{len(path_gene)}")
            pathway_graph = Cytograph.PathGraph(path_gene,max_hop)
            pathway_graph.built_edge(f_path)
            receptor_gene = pathway_graph.find_valid_lr_pair()
            print(f"receptor_gene---------{len(receptor_gene)}")
            lr_pair_sub = lr_pair_all[lr_pair_all['receptor'].isin(receptor_gene)]
            # lr_pair_sub.to_csv(f'{out_dir}/lr_pair_sub_{cell_sender}_{cell_receiver}.csv')

            print(f"lr_pair_sub---------{lr_pair_sub.shape}")

            
            if lr_pair_sub.shape[0] == 0:
                print (f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} because of no downstream transcriptional factors found for receptors!")
                continue
            else:
                print (f"the number of valid pathway number between {cell_sender} and {cell_receiver} is: {lr_pair_sub.shape[0]}")
                #find cell pair

                lr_pair_sub.insert(0, 'cell_sender', cell_sender)
                lr_pair_sub.insert(1, 'cell_receiver', cell_receiver)


                sig_lr_pair = find_sig_lr(spilt_dir,lr_pair_sub,cell_pair,gene_index_dict,cell_index_dict,per_num =1000,cells_per_file=cells_per_file)
                
                if sig_lr_pair.shape[0] == 0:
                    print (f"No ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression")
                    continue
                print (f"The ligand-recepotor pairs found between {cell_sender} and {cell_receiver} with significant expression is {sig_lr_pair.shape[0]}")

                find_significant_ligand_recepotor_pairs_time = datetime.datetime.now()
                print(f"find significant expression ligand-recepotor pairs time-------------{(find_significant_ligand_recepotor_pairs_time-find_ligand_recepotor_pairs_time).seconds}")
                
                tf = pathway_graph.find_lr_tf(sig_lr_pair)
                # tf.to_csv(f'{out_dir}/tf_{cell_sender}_{cell_receiver}.csv')

                find_tf_time = datetime.datetime.now()
                print(f"find tf time-------------{(find_tf_time-find_significant_ligand_recepotor_pairs_time).seconds}")

                print(f"tf------{tf.shape}")
                if(tf.shape[0]==0):
                    print(f"{cell_sender}_{cell_receiver} do not exist")
                    continue;
                path_score = get_score(sig_lr_pair,tf)

                print(f"path_score---{path_score.shape}")

                if path_score.shape[0] != 0:
                    all_lr_score=path_score


                obj['lr_score'] = all_lr_score
                obj['cell_pair'].update({f'{cell_sender}-{cell_receiver}': cell_pair})

                #检查目录是否存在
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)


                # 清理文件名中的特殊字符
                filename = f"cci_result_{cell_sender}_{cell_receiver}.pkl"
                filename = filename.replace("/", "_").replace(" ", "_")
                pickle_file_path = f"{out_dir}/{filename}"

                with open(pickle_file_path, 'wb') as pickle_file:
                    # data_to_save = {'cell_sender': cell_sender, 'cell_receiver': cell_receiver, 'sig_lr_pair': sig_lr_pair}
                    pickle.dump(obj, pickle_file,protocol=pickle.HIGHEST_PROTOCOL)
                    
                print(f"{cell_receiver} and {cell_sender} done. Data saved to {pickle_file_path}")

                # all_lr_score=pd.concat([all_lr_score,path_score],axis=0)
                # obj['cell_pair'].update({f'{cell_sender}-{cell_receiver}': cell_pair})
                # print(f"{cell_receiver} and {cell_sender} done")
        # obj['lr_score'] = all_lr_score
    else:
        if n_core > mp.cpu_count():
            n_core = mp.cpu_count()
        print(f"parallel processing with {n_core} cores")

        ###

        st_meta_1 = adata.obsm['meta']
        st_meta_1 = st_meta_1.drop(columns=['cell'])
        st_meta_1 = st_meta_1.reset_index()

        id_x_y_array = st_meta_1[['x', 'y']].values


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
        ###

        with mp.Pool(processes=n_core) as pool:
            results = pool.starmap_async(process_sender_receiver, [(i, lr_pair, distances_knn, indices_knn,sender_list, receiver_list, st_meta_1, pathway, valid_st_data, st_data, max_hop) for i in range(len(sender_list))])
            output = results.get()
            obj = post_process(output)

    with open(f'{out_dir}/cci_result.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    endtime = datetime.datetime.now()
    print(f"Total running time is {(endtime - starttime).seconds} seconds")


if __name__=="__main__":
    arguments = docopt(__doc__, version="CytoTour 1.0.0")
    main(arguments)




