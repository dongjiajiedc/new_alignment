import os
import numpy as np
import pandas as pd
import scib
import shutil
from core import *
import scanpy as sc
from pathlib import Path
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from os.path import exists
from alignment import *
from datasets.preprecossing import *
import plotly.graph_objects as go
from d3blocks import D3Blocks


def get_count_data(adata,counts_location=None):
    data = adata.layers[counts_location].copy() if counts_location else adata.X.copy()
    if not isinstance(data, np.ndarray):
        data= data.toarray()
    data_df = pd.DataFrame(data,index=adata.obs_names,columns=adata.var_names).transpose()
    return data_df


def check_paths(output_folder,output_prefix=None):
    output_path = os.path.join(os.getcwd(), output_folder)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    return output_path

def remove_batch_effect(pseudo_bulk, bulk_adata, out_dir, project='',batch_effect=True):
    """
    Remove batch effect between pseudo_bulk and input bulk data.

    Parameters
    ----------
    pseudo_bulk : anndata.AnnData
        An :class:`~anndata.AnnData` containing the pseudo expression.
    bulk_adata : anndata.AnnData
        An :class:`~anndata.AnnData` containing the input expression.
    out_dir : string, optional
        The path to save the output file.
    project : string, optional
        The prefix of output file.
        
    Returns
    -------
    Returns the expression after removing batch effect.

    """
    out_dir = check_paths(out_dir+'/batch_effect')
    if batch_effect:
        if exists(f'{out_dir}/{project}_batch_effected.txt'):
            print(f'{out_dir}/{project}_batch_effected.txt already exists, skipping batch effect.')
            bulk_data = pd.read_csv(f"{out_dir}/{project}_batch_effected.txt",sep='\t').T
        else:
            
            save=True
            # check path, file will be stored in out_dir+'/batch_effect'
            pseudo_bulk_df = get_count_data(pseudo_bulk)
            input_bulk_df= get_count_data(bulk_adata)

            bulk = pd.concat([pseudo_bulk_df,input_bulk_df], axis=1)

            cells = np.append(pseudo_bulk.obs_names, bulk_adata.obs_names)
            batch = np.append(np.ones((1,len(pseudo_bulk.obs_names))), np.ones((1,len(bulk_adata.obs_names)))+1)
            if save:
                bulk.to_csv(out_dir+f"/{project}_before_batch_effected.txt",sep='\t')
            meta = pd.DataFrame({"batch": batch,"cells":cells})
            # get r script path
            robjects.r.source('./combat.R')
            pandas2ri.activate()
            robjects.r.run_combat(bulk, meta, out_dir, project)
            # stop auto trans from pandas to r dataframe
            pandas2ri.deactivate()
            # add layer
            if exists(f'{out_dir}/{project}_batch_effected.txt'):
                bulk_data = pd.read_csv(f"{out_dir}/{project}_batch_effected.txt",sep='\t').T
            else:
                raise ValueError('The batch_effected data is not available')
        bulk_data.clip(lower=0,inplace=True)
        # print(pseudo_bulk)
        # print(pseudo_bulk.obs_names)
        pseudo_bulk.layers["batch_effected"] = bulk_data.loc[pseudo_bulk.obs_names,:].values
        bulk_adata.layers["batch_effected"] = bulk_data.loc[bulk_adata.obs_names,:].values
    else:
        pseudo_bulk_df = get_count_data(pseudo_bulk)
        input_bulk_df= get_count_data(bulk_adata)
        bulk = pd.concat([pseudo_bulk_df,input_bulk_df], axis=1)
        bulk.to_csv(out_dir+f"/{project}_batch_effected.txt",sep='\t')

    return pseudo_bulk,bulk_adata

def get_atc(ans,nodes1,adata1,adata2,inter_gene):
    anslist_dist = dict(ans)
    anslist_dist.keys()
    def search_lineage(now,path,anss):
        path.append(now.name)
        if(now.son==[]):
            anss.append(path);
            return
        
        for i in now.son:
            search_lineage(i,path.copy(),anss);
    temp1 = []
    search_lineage(nodes1[0],[],temp1)
    temp1
    route1 = []
    route2 = []

    for i in temp1:
        r1 = []
        r2 = []
        for j in i:
            if j in anslist_dist.keys():
                r1.append(j)
                r2.append(anslist_dist[j])
        route1.append(r1)
        route2.append(r2)
        
    adata1.obs.index = [i+'_1' for i in adata1.obs.index]
    adata2.obs.index = [i+'_2' for i in adata2.obs.index]
    score = [];

    for i,j in zip(route1,route2):
        print(i,j)
        # try:
        shutil.rmtree('./batch_effect/', ignore_errors=True)

        cells1 = [ ]
        cells2 = [ ]
        for k,t in zip(i,j):
            num1 = int(k.split('_')[0])
            num2 = int(t.split('_')[0])
            if(len(k.split('_'))>1):
                cells1.append(str(num1));
            if(len(t.split('_'))>1):
                cells2.append(str(num2));
        if(cells1==[] or cells2 ==[]):
            continue;
        
        sub_adata1 = adata1[adata1.obs['leiden'].isin(cells1),inter_gene].copy();
        sub_adata2 = adata2[adata2.obs['leiden'].isin(cells2),inter_gene].copy();

        adata1_after,adata2_after = remove_batch_effect(sub_adata1.copy(),sub_adata2.copy(),'./')
        
        sc.pp.neighbors(sub_adata1,use_rep='X')
        sc.tl.diffmap(sub_adata1)
        sub_adata1.uns['iroot'] = 0
        sc.tl.dpt(sub_adata1)
        
        
        sc.pp.neighbors(sub_adata2,use_rep='X')
        sc.tl.diffmap(sub_adata2)
        sub_adata2.uns['iroot'] = 0
        sc.tl.dpt(sub_adata2)
        
        
        adata1_after.obsm['batch_effected'] = adata1_after.layers['batch_effected']
        adata2_after.obsm['batch_effected'] = adata2_after.layers['batch_effected']
        
        sc.pp.neighbors(adata1_after,use_rep='batch_effected')
        sc.tl.diffmap(adata1_after)
        adata1_after.uns['iroot'] = 0
        sc.tl.dpt(adata1_after)

        sc.pp.neighbors(adata2_after,use_rep='batch_effected')
        sc.tl.diffmap(adata2_after)
        adata2_after.uns['iroot'] = 0
        sc.tl.dpt(adata2_after)
        
        score.append( scib.me.trajectory_conservation(sub_adata1, adata1_after, label_key="celltype"))
        score.append( scib.me.trajectory_conservation(sub_adata2, adata2_after, label_key="celltype"))
        shutil.rmtree('./batch_effect/', ignore_errors=True)
        # except:
        #     print(i,j)

        #     pass;
    return np.array(score).mean()

# Load d3blocks
def chord_graph(ans,nodes1,nodes2):
    def cost(i,j):
        df = pd.DataFrame(
            {"A": i.value, "B":j.value})
        mincost = df.corr(method="spearman").iloc[0, 1] +1
        return mincost/2
    l1=[]
    l2=[]
    l3=[]
    for i,j in ans:
        index1 = [k for k in nodes1 if(str(k)==i)][0]
        index2 = [k for k in nodes2 if(str(k)==j)][0]
        l1.append('1_'+i)
        l2.append('2_'+j)
        l3.append(cost(index1,index2))

    d3 = D3Blocks(chart='Chord', frame=False)
    # Load example data
    df=pd.DataFrame(
        {"source":l1,"target":l2,'weight':l3}
    )


    d3.set_node_properties(df)
    d3.set_edge_properties(df)
    for i in range(len(df)):
        d3.node_properties.get(df.iloc[i][0])['color']='#949398FF'
        d3.node_properties.get(df.iloc[i][1])['color']='#F4DF4EFF'
    d3.show()


def show_3d(ans,nodes1,nodes2):
    t=show_graph(ans,nodes1[0],nodes2[0]);
    s = 13
    # Helix equation
    x, y, z =t.pos_x,t.pos_y,[0 for i in t.pos_x]
    names = [i+' '+j for i,j in zip(t.labels,t.hover_text)]
    
    x=np.array(x)
    y=np.array(y)
    layout = go.Layout(
        scene=dict(
            zaxis=dict(
                range=[0, 1]  # Set the desired z-axis range
            )
        ),    
    )
    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(size=3),hoverinfo='text',hovertext=names)],layout=layout)

    x, y, z =t.pos_x_2,t.pos_y_2,[0.5 for i in t.pos_x_2]
    x=np.array(x)-s
    y=np.array(y)
    names = [i+' '+j for i,j in zip(t.labels_2,t.hover_text_2)]

    fig.add_trace(go.Scatter3d(x=x, y=y, z=z,mode='markers',marker=dict(size=3),hoverinfo='text',hovertext=names))

    for i in t.edges:
        p1,p2 = i;
        index1=t.label_hash[p1];
        index2=t.label_hash[p2];
        
        x1=t.pos_x[index1];y1=t.pos_y[index1];
        x2=t.pos_x[index2];y2=t.pos_y[index2];
        fig.add_trace(go.Scatter3d(x=[x1,x2], y=[y1,y2], z=[0,0],hoverinfo='none',mode='lines',line=dict(
                        color="#333",
                        width=4,
                    ),))
    for i in t.edges_2:
        p1,p2 = i;
        index1=t.label_hash_2[p1];
        index2=t.label_hash_2[p2];
        
        x1=t.pos_x_2[index1]-s;y1=t.pos_y_2[index1];
        x2=t.pos_x_2[index2]-s;y2=t.pos_y_2[index2];
        fig.add_trace(go.Scatter3d(x=[x1,x2], y=[y1,y2], z=[0.5,0.5],hoverinfo='none',mode='lines',line=dict(
                        color="#acd",
                        width=4,
                    ),))
        for i in t.ans:
            p1,p2 = i;
            index1=t.label_hash[p1];
            index2=t.label_hash_2[p2];
            x1=t.pos_x[index1];y1=t.pos_y[index1];
            x2=t.pos_x_2[index2]-s;y2=t.pos_y_2[index2];
            fig.add_trace(go.Scatter3d(x=[x1,x2], y=[y1,y2], z=[0,0.5],hoverinfo='none',mode='lines',line=dict(
                        color="#345681",
                        width=4,
                    ),))
    fig.update_layout(showlegend=False)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    fig.show()
def show_2d(ans,nodes1,nodes2):
    show_graph(ans,nodes1[0],nodes2[0],color=['#184e77','#1a759f','#168aad',"#34a0a4",'#52b69a','#99d98c','#76c893','#99d98c']).show_fig()