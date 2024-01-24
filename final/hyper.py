import logging
import os
import numpy as np
import torch
import torch.utils.data as data

import optim
from datasets.hc_dataset import HCDataset
from datasets.balance_dataset import balance_dataset
from datasets.loading import load_data
from model.hyphc import HypHC
from model.balancehc import balancehc
from utils.mst import mst
from utils.metrics import dasgupta_cost
from utils.unionfind import UnionFind
import networkx as nx

from utils.poincare import project,hyp_dist
from utils.lca import hyp_lca

from alignment import *



from utils.lca import hyp_lca

def hyp_lca_numpy(x, y):
    """
    Computes the hyperbolic LCA in numpy.
    """
    x = torch.from_numpy(x).view((1, 2))
    y = torch.from_numpy(y).view((1, 2))
    lca = hyp_lca(x, y, return_coord=True)
    return lca.view((2,)).numpy()
def is_leaf(tree, node):
    """
    Check if node is a leaf in tree.
    """
    return len(list(tree.neighbors(node))) == 0
def complete_tree(tree, leaves_embeddings):
    """
    Get embeddings of internal nodes from leaves' embeddings using LCA construction.
    """

    def _complete_tree(embeddings, node):
        children = list(tree.neighbors(node))
        if len(children) == 2:
            left_c, right_c = children
            left_leaf = is_leaf(tree, left_c)
            right_leaf = is_leaf(tree, right_c)
            if left_leaf and right_leaf:
                pass
            elif left_leaf and not right_leaf:
                embeddings = _complete_tree(embeddings, right_c)
            elif right_leaf and not left_leaf:
                embeddings = _complete_tree(embeddings, left_c)
            else:
                embeddings = _complete_tree(embeddings, right_c)
                embeddings = _complete_tree(embeddings, left_c)
            embeddings[node] = hyp_lca_numpy(embeddings[left_c], embeddings[right_c])
        return embeddings

    n = leaves_embeddings.shape[0]
    tree_embeddings = np.zeros((2 * n - 1, 2))
    tree_embeddings[:n, :] = leaves_embeddings
    root = max(list(tree.nodes()))
    tree_embeddings = _complete_tree(tree_embeddings, root)
    return tree_embeddings

def train(model,dataloader,optimizer,similarities,epoches):
    """
    Train the embedding model
    """

    best_cost = np.inf
    best_model = None
    counter = 0
    for epoch in range(epoches):
        model.train()
        total_loss = 0.0
        for step, (triple_ids, triple_similarities) in enumerate(dataloader):
            loss = model.loss(triple_ids, triple_similarities)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        total_loss = total_loss / (step + 1.0)
        print("\t Epoch {} | average train loss: {:.6f}".format(epoch, total_loss))

        if (epoch + 1) % 1 == 0:
            tree = model.decode_tree(fast_decoding=1)
            cost = dasgupta_cost(tree, similarities)
            logging.info("{}:\t{:.4f}".format("Dasgupta's cost", cost))
            if cost < best_cost:
                counter = 0
                best_cost = cost
                best_model = model.state_dict()
            else:
                counter += 1
                if counter == 20:
    #                 logging.info("Early stopping.")
                    return

    # anneal temperature
        if (epoch + 1) % 30 == 0:
            model.anneal_temperature(0.5)
    #         logging.info("Annealing temperature to: {}".format(model.temperature))
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                lr = param_group['lr']
            print("Annealing learning rate to: {}".format(lr))

        print("Optimization finished.")
        if best_model is not None:
            # load best model
            model.load_state_dict(best_model)
            
def train2(model,dataloader,optimizer,epoches):
    """
    Train the rotation model
    """
    best_cost = np.inf
    best_model = None
    counter = 0
    for epoch in range(epoches):
        model.train()
        total_loss = 0.0
        for step, datas in enumerate(dataloader):
            loss = model.loss(datas[0],datas[1],datas[2],datas[3],datas[4],datas[5])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        total_loss = total_loss / (step + 1.0)
        print("\t Epoch {} | average train loss: {:.6f}".format(epoch, total_loss))
        model.update();
        
        cost = total_loss;
        if cost < best_cost:
            counter = 0
            best_cost = cost
            best_model = model.state_dict()
        else:
            counter += 1
            if counter == 20:
#                 logging.info("Early stopping.")
                return
            
    if best_model is not None:
        # load best model
        model.load_state_dict(best_model)
                

def sl_np_mst_ij(xs, S):
    """
    Return the ij to merge the unionfind
    """
    xs = project(xs).detach()
    xs0 = xs[None, :, :]
    xs1 = xs[:, None, :]
    sim_mat = S(xs0, xs1)  # (n, n)
    similarities = sim_mat.numpy()
    n = similarities.shape[0]
    similarities=similarities.astype('double')
    ij, _ = mst(similarities, n)
    return ij
  
def get_colors(y, color_seed=1234):
    """
    Random color assignment for label classes.
    """
    np.random.seed(color_seed)
    colors = {}
    for k in np.unique(y):
        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        colors[k] = (r, g, b)
    return [colors[k] for k in y]


def search_merge_tree(now,ids,save_path,values,fathers,xys):
    """
    Search the tree and save the information
    """
    fathers.append(ids);
    values.append(now.name);
    xys.append(now.value);
    now_id = len(values)-1;
    for son in now.son:
        search_merge_tree(son,now_id,save_path,values,fathers,xys)

def deep_search_tree(now,depth,path,f):
    """
    Search the tree and calculate the information
    """
    now.f=f
    now.depth=depth;
    path.append(now);
    now.path=path.copy();
    if(f!=now):
        now.distance_to_root = f.distance_to_root + hyp_dist(f.value,now.value)
    else:
        now.distance_to_root = 0
        
    for i in now.son:
        deep_search_tree(i,depth+1,path,now);
        
    path.remove(now)

def get_Hyper_tree(data_path,start,end,label,epoches,model_path=None,model_path2=None,save_path='./'):
    """
    Embedding the dataset into hyperbolic tree structure
    
    Parameters
    ----------
    data_path : string
        Path of the cluster center file
    start : int
        Index of the starting in the data file
    end : int
        Index of the ending in the data file
    label : int
        Index of the label in the data file
    epoches : int
        Number of epochs for hyper-embedding model of the first dataset
    model_path : string
        Path to the model of the embedding
    model_path2 : string
        Path to the model of the rotating
    save_path : string 
        Path to the folder to save the data files

    Returns
    -------
    Save the training models and tree structure files in the folder

    """
    
    np.random.seed(1234)
    torch.manual_seed(1234)
    x, y_true, similarities = load_data(data_path,start,end,label)
    print("{} length:{}".format(data_path,len(y_true)));
    dataset = HCDataset(x, y_true, similarities, num_samples=50000)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)
    model = HypHC(dataset.n_nodes, 2, 5e-2, 5e-2 ,0.999)

    if(model_path==None or os.path.exists(model_path)==False):
        model.to("cpu")
        Optimizer = getattr(optim, 'RAdam')
        optimizer = Optimizer(model.parameters(),0.0005)
        train(model,dataloader,optimizer,similarities,epoches);
        torch.save(model.state_dict(),save_path+'model.pth');
    else:
        params = torch.load((model_path), map_location=torch.device('cpu'))
        model.load_state_dict(params, strict=False)

    model.eval()
    
    sim_fn = lambda x, y: torch.sum(x * y, dim=-1)
    n=len(x);
    leaves_embeddings = model.normalize_embeddings(model.embeddings.weight.data)
    leaves_embeddings = project(leaves_embeddings).detach().cpu()
    ijs = sl_np_mst_ij(leaves_embeddings,sim_fn)
    uf = UnionFind(n)
    uf.merge(ijs)

    tree = nx.DiGraph()
    for i, j in enumerate(uf.tree()[:-1]):
        if(j!=-1):
            tree.add_edge(j, i)

    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(111)
    # circle = plt.Circle((0, 0), 20.0, color='r', alpha=0.1)
    # ax.add_artist(circle)
    
    n = len(leaves_embeddings)
    embeddings = complete_tree(tree, leaves_embeddings)
    
    # where_are_NaNs = np.isnan(embeddings)
    # embeddings[where_are_NaNs] = 0
    # colors = get_colors(y_true, 1234)
    # ax.scatter(embeddings[:n, 0]*20, embeddings[:n, 1]*20, c=colors, s=50, alpha=0.6)
    # for i in range(len(embeddings)):
    #     if(i<n):
    #         continue;
    #     if(uf.mer[i]!=-1):
    #         pass;
    #     else:
    #         ax.scatter(embeddings[i][0]*20,embeddings[i][1]*20,color='black',s=20,alpha=0.7)
    # for n1, n2 in tree.edges():
    #     x1 = embeddings[n1];
    #     x2 = embeddings[n2];
    #     plot_geodesic(x1,x2,ax)
    # fig.savefig(save_path+"graph.png");
    # embeddings = np.array(uf.pos)
    

    nodes1 = [node(name=str(i),son=[]) for i in range(len(uf.tree()))]
    for i in range(n):
        nodes1[i].subson=[i];
    for i,j in enumerate(uf.tree()):
        if(j!=-1):
            nodes1[j].son.append(nodes1[i])
        nodes1[i].value=torch.tensor(embeddings[i]);
        nodes1[j].subson.extend(nodes1[i].subson)
    root = nodes1[-1];

    values = [];
    fathers = [];
    xys = [];
    search_merge_tree(root,-1,0,values,fathers,xys)
    np.save(save_path+"dataname.npy",values)
    np.save(save_path+"datalink.npy",fathers)
    np.save(save_path+"dataxy.npy",[i.numpy() for i in xys])

    deep_search_tree(root,0,[],root)
    result = []
    distances = []
    for index in range(n,len(nodes1)):
        i = nodes1[index]
        if(len(i.subson)==2):
            for j in i.rest(n):
                result.append([i.subson,j,int(i),1,int(j)])
                
    for index in range(n,len(nodes1)-1):
        i = nodes1[index]
        for i1 in range(len(i.subson)):
            for i2 in range(i1+1,len(i.subson)):
                for j in i.rest(n):
                    result.append([[i.subson[i1],i.subson[i2]],j,int(i),0,int(i.f)])
            
    for i in nodes1:
        distances.append(i.distance_to_root);
        
    distances = torch.tensor(distances)
    embeddings = complete_tree(tree, leaves_embeddings)
    embeddings = torch.tensor(embeddings)


    dataset_test = balance_dataset(similarities,100,embeddings,distances,result)
    dataloader = data.DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)


    model2 = balancehc(nodes1,torch.tensor(embeddings),hyperparamter = 1)
                
    if(model_path2==None or os.path.exists(model_path2)==False):
        Optimizer = getattr(optim, 'RAdam')
        optimizer = Optimizer(model2.parameters(),0.0005)
        train2(model2,dataloader,optimizer,epoches)
        torch.save(model2.state_dict(),save_path+'balance_model.pth');
    else:
        params = torch.load((model_path2), map_location=torch.device('cpu'))
        model2.load_state_dict(params, strict=False)
        
    model2.eval()

    temp = model2.embeddings.weight.data
    after_balance = embeddings.numpy().copy();
    for i in range(len(temp)):
        after_balance[i] = temp[i].detach().numpy() 
    # after_balance = model2.normalize_embeddings(torch.tensor(after_balance))
    
    after_balance = project(torch.tensor(after_balance))
    after_balance = np.array(after_balance)
    
    # colors = get_colors(y_true, 1234)
    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(111)
    # circle = plt.Circle((0, 0), 20.0, color='r', alpha=0.1)
    # ax.add_artist(circle)
    # ax.scatter(after_balance[:n, 0]*20, after_balance[:n, 1]*20, c=colors, s=50, alpha=0.6)
    # ax.scatter(after_balance[n:,0]*20,after_balance[n:,1]*20,color ='black',s=20,alpha=0.6)
    # for n1, n2 in tree.edges():
    #     x1 = after_balance[n1];
    #     x2 = after_balance[n2]
    #     plot_geodesic(x1,x2,ax)
    # fig.savefig(save_path+"graph_after.png")

    np.save(save_path+'dataxy.npy',np.array([after_balance[j] for j in [int(i) for i in values]]))

    # return loss2