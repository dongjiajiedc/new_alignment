
import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import itertools
import pulp as plp
from utils.poincare import hyp_dist
import torch
import random
import os
import glob


class node:
    """
    Class of the node of the tree
    """
    def __init__(self,value=None,son=[],name='',hyper=None):
        self.value = value;
        self.hyper = hyper;
        self.son = son;
        self.name =name;
        self.f = None;
        self.depth=0;
        self.subson= [];
        self.distance_to_root=0;
        self.path=None;
        self.num_son=0;
        self.dfs=None;
    def __int__(self):
        return int(self.name);
    def __repr__(self):
        return self.name
    def __str__(self):
        return self.name
    def copy(self):
        return node(self.value,self.son,self.name,self.hyper)
    def rest(self,n):
        all = [i for i in range(n)];
        result = [element for element in all if element not in self.subson];
        return result
    
class newnode:
    """
    Class of the aligned nodes by linear programming
    """
    def __init__(self,node1,node2):
        self.node1 = node1
        self.node2 = node2
        self.f = None
        self.edge = [];
        self.indegree = 0;
        self.value = 0;
        self.name = self.node1.name + self.node2.name
    def __str__(self):
        return "{}_{}".format(self.node1,self.node2)
    def __repr__(self):
        return "{}_{}".format(self.node1,self.node2)

        
    # def __lt__(self, other):
    #     return self.depth < other.depth
    
class tree_alignment:
    """
    Class is used to perform tree alignment between two trees
    """
    def __init__(self,root1,root2,cost1,lr=False,datas1=None,datas2=None):
        self.cost1 = cost1;
        self.dp = dict();
        self.forestdp = dict();
        self.anslist = [];
        self.ansnodes = [];
        self.lr = lr;
        self.lrcost = dict();
        self.root1 = root1;
        self.root2 = root2;
        self.minn = math.inf;
        self.cal_depth(root1,0)
        self.cal_depth(root2,0)
        self.datas1 = datas1
        self.datas2 = datas2
        pass;

    def cal_depth(self, now,d):
        """
        Calculate the depth of each node
        """
        now.depth = d;
        for i in now.son:
            self.cal_depth(i,d+1);
            
    def cal_lr_cost(self,t1,t2):
        
        
        files = glob.glob('./lr/after/datas/temp/*')
        for f in files:
            os.remove(f)
        tdata1 = self.datas1[self.datas1.obs[self.datas1.obs['leiden'].isin([str(i) for i in t1.subson])].index]
        tdata2 = self.datas2[self.datas2.obs[self.datas2.obs['leiden'].isin([str(i) for i in t2.subson])].index]
        tdatas = tdata1.concatenate(tdata2)
        tdatas.write_h5ad('./lr/after/datas/temp/AN.h5ad')

        def lr_s(n):
            a1= 'python3 lr/after/splith5ad.py lr/after/datas/{}/AN.h5ad lr/after/datas/{} lr/after/datas/pathways.tsv 1000'.format(n,n)
            a2 = 'python3 lr/after/CytoTour.py lr/after/datas/lr_database.csv lr/after/datas/pathways.tsv lr/after/datas/{} 1000 lr/after/datas/{}/AN.h5ad --out lr/after/datas/{}'.format(n,n,n)
            return a1,a2;
        a1,a2 = lr_s('temp')
        try:
            os.system(a1)
            print("finish 1")
            os.system(a2)
            print("finish 2")        
            df = pd.read_pickle('./lr/after/datas/temp/cci_result_1.0_2.0.pkl')
            print(t1,t2,'right')

            return df['lr_score']['score'].mean()
        
        except:
            print(t1,t2,'error')
            return 0.01
        
        # t1 = str((1+int(t1)/100))
        
        # t2 = str((2+int(t2)/100))
        
        
        
        # df = './lr/after/datas/temp/cci_result_{}_{}.pkl'.format(t1,t2);
        # df2 = './lr/after/datas/temp/cci_result_{}_{}.pkl'.format(t2,t1);
        # if(os.path.isfile(df)):
        #     t = pd.read_pickle(df)
        #     return t['lr_score']['score'].mean()
        #     pass;
        # elif(os.path.isfile(df2)):
        #     t = pd.read_pickle(df2)
        #     return t['lr_score']['score'].mean()
        # else:
        #     return 0.01;
        
    def cost(self,i,j):
        """
        Cost of aligning two nodes
        """
        if(i==[] and j==[]):
            return 0;
        if(i==[]):
            # return (j.r-j.l)**2;
            # return np.linalg.norm(j.value-self.cost1);
            return self.cost1

        if(j==[]):
            # return (i.r-i.l)**2;
            # return np.linalg.norm(i.value-self.cost1);
            return self.cost1
        # if(i ==[] or j==[]):
        #     return self.cost1;
        else:
            # return (i.l-j.l)**2 + (i.r-j.r)**2;
            if(self.lr== False):
                df = pd.DataFrame(
                    {"A": i.value, "B":j.value})
                mincost = 1 - df.corr(method="spearman").iloc[0, 1]
            else:
                if(self.lrcost.__contains__((i.name,j.name))):
                    pass;
                else:
                    self.lrcost[(i.name,j.name)] = self.cal_lr_cost(i,j);
                mincost = self.lrcost[(i.name,j.name)];
            return mincost
            # return np.linalg.norm(i.value-j.value);
    
    def cal_tree(self,i,j):
        """
        Cost of aligning two trees
        """
        if(i==[] and j==[]):
            return 0;
        
        if(j==[]):
            
            if(self.dp.__contains__((i.name,"Empty"))):
                pass;
            else:
                self.dp[(i.name,"Empty")]=self.cost(i,[]) + self.cal_forest(i.son,[]);
                
            return self.dp[(i.name,"Empty")];
        
        
        if(i==[]):
            
            if(self.dp.__contains__(("Empty",j.name))):
                    pass;
            else:
                self.dp[("Empty",j.name)]=self.cost([],j) + self.cal_forest([],j.son);
                
            return self.dp[("Empty",j.name)];
        
        
        
        if(self.dp.__contains__((i.name,j.name))):
            pass;
        
        else:
            temp = self.cal_forest(i.son,j.son) + self.cost(i,j);

            temp2 = self.cal_tree([],j);
            
            if(j.son==[]):
                temp3 = temp2 + self.cal_tree(i,[]) - self.cal_tree([],[])

                if(temp >temp3):
                    temp = temp3;
                    # ans = (i.name,"Empty");

            for son in j.son:
                temp3 = temp2 + self.cal_tree(i,son) - self.cal_tree([],son)
                # temp = min(temp, temp3)

                if(temp>temp3):
                    temp = temp3;
                    # ans = (i.name,son.name);
                # t= self.cal_tree(i,son) - self.cal_tree([],son);
                # temp3 = min(temp3,t);
            # temp =min(temp,temp3+self.cal_tree([],j));
            # temp2 = 100000;
            
            temp2 = self.cal_tree(i,[]);
            
            if(i.son==[]):
                temp3 = temp2+self.cal_tree([],j) - self.cal_tree([],[]);
                # temp = min(temp,temp3)
                if(temp>temp3):
                    temp = temp3;
                    # ans = ("Empty",j.name);
                
            for son in i.son:
                temp3= temp2 + self.cal_tree(son,j) - self.cal_tree(son,[]);
                # temp = min(temp, temp3)

                if(temp>temp3):
                    temp = temp3;
                    # ans = (son.name,j.name);
                    
                    
            #     t = self.cal_tree(son,j) - self.cal_tree(son,[]);
            #     temp2 = min(temp2,t);
            # temp = min(temp,temp2 + self.cal_tree(i,[]));
            # self.anslist.append(ans);
            self.dp[(i.name,j.name)]= temp;
            
        return self.dp[(i.name,j.name)];
    
    def cal_forest(self,i,j):
        """
        Cost of aligning two forests
        """
        if(i==[] and j==[]):
            return 0;
        
        if(i==[]):
            if(self.forestdp.__contains__(("Empty",str(j)))):
                pass;
            else:
                sum=0;
                for son in j:
                    sum+= self.cal_tree([],son);
                    
                self.forestdp[("Empty",str(j))] = sum;
                
            return self.forestdp[("Empty",str(j))];
            
        
        
        
        if(j==[]):
            if(self.forestdp.__contains__((str(i),"Empty"))):
                    pass;
            else:
                sum=0;
                for son in i:
                    sum+= self.cal_tree(son,[]);
                    
                self.forestdp[(str(i),"Empty")] = sum;
                
            return self.forestdp[(str(i),"Empty")];
            # sum=0;
            # for son in i:
            #     sum+= self.cal_tree(son,[]);
            # return sum;
        
        
        # leni = len(i);
        # lenj = len(j);
        # temp = min (temp,self.cal_forest(i[0:leni-1],j[0:lenj-1])+ self.cal_tree(i[leni-1],j[lenj-1]));
        # temp = min (temp,self.cal_forest(i,j[0:lenj-1])+ self.cal_tree([],j[lenj-1]));
        # temp = min (temp,self.cal_forest(i[0:leni-1],j)+ self.cal_tree(i[leni-1],[]));
        
        # temp2 = self.cost([],j[lenj-1]);
        
        # for m in range(1,leni):
        #     temp = min(temp, temp2 + self.cal_forest(i[0:m],j[0:lenj-1]) + self.cal_forest(i[m:leni],j[lenj-1].son));
            
        # temp3 = self.cost(i[leni-1],[]);
        
        # for m in range(1,lenj):
        #     temp = min(temp, temp3 + self.cal_forest(i[0:leni-1],j[0:m]) + self.cal_forest(i[leni-1].son,j[m:lenj]));
        if(self.forestdp.__contains__((str(i),str(j)))):
            pass;
        else:
            temp = math.inf;

            for l in i:
                for r in j:
                    i1 = i.copy();
                    j1 = j.copy();
                    i1.remove(l);
                    j1.remove(r);
                    temp = min(temp,self.cal_forest(i1,j1) + self.cal_tree(l,r));
                    
            for l in i:
                for m in range(1, len(j)+1):
                    for r in list(itertools.combinations(j, m)):
                        i1 = i.copy();
                        j1 = j.copy();
                        i1.remove(l);
                        temp = min(temp,self.cal_forest(l.son,list(r))+self.cal_forest(i1,list(set(j1)-set(r)))+self.cost(l,[]))
                        # temp = min(temp,self.cal_forest(l.son,list(r))+self.cal_forest(i1,list(set(j1)-set(r)))+ 1 )
            # for l in i:
            #     i1 = i.copy();
            #     j1 = j.copy();
            #     i1.remove(l);
            #     temp =  min(temp,self.cal_forest(l.son,[])+self.cal_forest(i1,j1)+self.cost(l,[]))
                
            # for m in range(1, len(j)):
            #     for r in list(itertools.combinations(j, m)):
            #         i1 = i.copy();
            #         j1 = j.copy();
            #         # i1.remove(l);
            #         temp = min(temp,self.cal_forest([],list(r))+self.cal_forest(i1,list(set(j1)-set(r))))
                    
                    # temp = min(temp,self.cal_forest(l.son,list(r))+self.cal_forest(i1,list(set(j1)-set(r)))+ 1 )
                
            for r in j:
                for m in range(1, len(i)+1):
                    for l in list(itertools.combinations(i, m)):
                        i1 = i.copy();
                        j1 = j.copy();
                        j1.remove(r);

                        temp = min(temp,self.cal_forest(list(l),r.son)+self.cal_forest(list(set(i1)-set(l)),j1)+self.cost([],r))
                        # temp = min(temp,self.cal_forest(list(l),r.son)+self.cal_forest(list(set(i1)-set(l)),j1)+ 1)
                        
            # for m in range(1, len(i)):
            #     for l in list(itertools.combinations(i, m)):
            #         i1 = i.copy();
            #         j1 = j.copy();
            #         # j1.remove(r);
            #         temp = min(temp,self.cal_forest(list(l),[])+self.cal_forest(list(set(i1)-set(l)),j1))
                    
            # for r in j:
            #     i1 = i.copy();
            #     j1 = j.copy();
            #     j1.remove(r);
            #     temp  = min(temp, self.cal_forest([],r.son)+ self.cal_forest(i1,j1)+self.cost([],r));
                    
            self.forestdp[(str(i),str(j))]=temp;
        return self.forestdp[(str(i),str(j))];
    
    def search_alignment_tree(self,i,j,value):
        """
        Find the mininum cost of two trees
        """
        if(i==[] or j==[]):
            return ;
        temp1 = self.cal_forest(i.son,j.son)
        temp = self.cal_forest(i.son,j.son) + self.cost(i,j);
        
        if(temp == value):
            self.anslist.append((i.name,j.name));
            self.ansnodes.append((i,j));
            self.search_alignment_forest(i.son,j.son,temp1);
            return;
            
        temp2 = self.cal_tree([],j);
        
        if(j.son==[]):
            temp3 = temp2 + self.cal_tree(i,[]) - self.cal_tree([],[])
            if(temp3 == value):
                # self.anslist.append("Empty",j.name);
                # self.anslist.append(i.name,"Empty");
                return;
            
        for son in j.son:
            temp3 = temp2 + self.cal_tree(i,son) - self.cal_tree([],son)
            if(temp3==value):
                # self.anslist.append("Empty",j.name);
                self.search_alignment_tree(i,son, self.cal_tree(i,son));
                return;
                # self.anslist.append(i,son);

        temp2 = self.cal_tree(i,[]);
        
        if(i.son==[]):
            temp3 = temp2+self.cal_tree([],j) - self.cal_tree([],[]);
            
            if(temp3 == value):
                # self.anslist.append("Empty",j.name);
                # self.anslist.append(i.name,"Empty");
                return;
            
        for son in i.son:
            temp3= temp2 + self.cal_tree(son,j) - self.cal_tree(son,[]);
            
            if(temp3 == value):
                # self.anslist.append(i.name,"Empty");
                self.search_alignment_tree(son,j,self.cal_tree(son,j));
                return;
        
        return;

    def search_alignment_forest(self,i,j,value):
        """
        Find the mininum cost of two forests
        """
        if(i==[] or j==[]):
            return;
        for l in i:
            for r in j:
                i1 = i.copy();
                j1 = j.copy();
                i1.remove(l);
                j1.remove(r);

                temp1 = self.cal_forest(i1,j1);
                if(self.cal_forest(i1,j1)+ self.cal_tree(l,r) == value):
                    self.search_alignment_forest(i1,j1,temp1);
                    self.search_alignment_tree(l,r,self.cal_tree(l,r));
                    return
        for l in i:
            for m in range(1, len(j)+1):
                for r in list(itertools.combinations(j, m)):
                    i1 = i.copy();
                    j1 = j.copy();
                    i1.remove(l);
                    temp1 = self.cal_forest(l.son,list(r));
                    temp2 = self.cal_forest(i1,list(set(j1)-set(r)));
                    if(temp1 + temp2 +self.cost(l,[]) == value):
                    # if(temp1 + temp2 +1 == value):

                        self.search_alignment_forest(l.son,list(r),temp1);
                        self.search_alignment_forest(i1,list(set(j1)-set(r)),temp2);
                        return
                    # temp = min(temp,self.cal_forest(l.son,list(r))+self.cal_forest(i1,list(set(j1)-set(r)))+self.cost(l,[]))
        for r in j:
            for m in range(1, len(i)+1):
                for l in list(itertools.combinations(i, m)):
                    i1 = i.copy();
                    j1 = j.copy();
                    j1.remove(r);
                    temp1 = self.cal_forest(list(l),r.son)
                    temp2 = self.cal_forest(list(set(i1)-set(l)),j1);
                    # if(self.cal_forest(list(l),r.son)+self.cal_forest(list(set(i1)-set(l)),j1)+self.cost([],r) == value):
                    if(temp1+temp2+ self.cost([],r) == value):

                        self.search_alignment_forest(list(l),r.son,temp1);
                        self.search_alignment_forest(list(set(i1)-set(l)),j1,temp2);
                        return;
                    # temp = min(temp,self.cal_forest(list(l),r.son)+self.cal_forest(list(set(i1)-set(l)),j1)+self.cost([],r))

        return ;
    def printdp(self):
        temp =list(self.dp.keys());
        # temp.sort()
        j=temp[0][0];
        
        for i in temp:
            if(i[0]!=j):
                print();j=i[0];
            print("{}={}".format(i,self.dp[i]),end=" ");
            
    def get_dp(self):
        return self.dp;
    
    def get_ans(self):
        return self.anslist;
    
    def run_alignment(self):
        self.minn = self.cal_tree(self.root1,self.root2);
        self.search_alignment_tree(self.root1,self.root2,self.minn);
        return self.minn;
    
    def show_ans(self):
        print("The mininum cost for alignment is {}".format(self.minn));
        print("The alignment edges list is {}".format(self.anslist));
        # print("The dp result");
        # self.printdp();
        # print(self.anslist);
# class show_graph:
#     def __init__(self,ans,root1,root2):
#         self.ans = ans;
        
#         self.pos_x=[];
#         self.pos_y=[];
#         self.edges=[];
#         self.label_hash = dict(); 
#         self.labels = [];
#         self.hover_text =[];
#         self.values=[];
#         self.cnt = 0;
        
#         self.pos_x_2=[];
#         self.pos_y_2=[];
#         self.edges_2=[];
#         self.label_hash_2 = dict(); 
#         self.labels_2 = []
#         self.hover_text_2 = []
#         self.values_2=[];
#         self.cnt2 = 0;
#         self.fig = go.Figure();
        
#         self.root1 = root1;
#         self.root2 = root2;
#         self.height = 5;
#         self.run_graph();
        
#     def cal_tree_pos(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values):
#         mid = (l+r)/2
#         pos_x.append(mid);
#         pos_y.append(h)
#         label_hash[now.name]=self.cnt;
#         num_son = len(now.son);
#         labels.append("{}".format(now.name));
#         edges.append((f.name,now.name));
#         hover_text.append("connect Empty");
#         values.append(np.linalg.norm(now.value));
#         if(num_son == 0):
#             return;
#         length = (r-l)/num_son;
#         for i in range(num_son):
#             self.cnt+=1;
#             self.cal_tree_pos(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values);
            
#     def cal_tree_pos2(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values):
#         mid = (l+r)/2
#         pos_x.append(mid);
#         pos_y.append(h)
#         label_hash[now.name]=self.cnt2;
#         num_son = len(now.son);
#         labels.append("{}".format(now.name));
#         edges.append((f.name,now.name));
#         hover_text.append("connect Empty");
#         values.append(np.linalg.norm(now.value))
#         if(num_son == 0):
#             return;

#         length = (r-l)/num_son;
#         for i in range(num_son):
#             self.cnt2+=1;
#             self.cal_tree_pos2(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values);
            
#     def run_graph(self):
        
#         self.cal_tree_pos(self.root1,1,20,self.height,self.root1, self.pos_x,self.pos_y,self.edges,self.label_hash,self.labels,self.hover_text,self.values);
        
#         self.cal_tree_pos2(self.root2,18,37,self.height-1.5,self.root2, self.pos_x_2,self.pos_y_2,self.edges_2,self.label_hash_2,self.labels_2,self.hover_text_2,self.values_2);
        
#         for i in self.edges:
#             p1,p2 = i;
#             index1=self.label_hash[p1];
#             index2=self.label_hash[p2];

#             x1=self.pos_x[index1];y1=self.pos_y[index1];
#             x2=self.pos_x[index2];y2=self.pos_y[index2];
#             self.fig.add_shape(
#                 type="line",
#                 x0=x1, y0=y1, x1=x2, y1=y2,
#                 line=dict(
#                     color="#333",
#                     width=4,
#                 ),
#                 layer="below"
#             )

#         for i in self.edges_2:
#             p1,p2 = i;

#             index1=self.label_hash_2[p1];
#             index2=self.label_hash_2[p2];

#             x1=self.pos_x_2[index1];y1=self.pos_y_2[index1];
#             x2=self.pos_x_2[index2];y2=self.pos_y_2[index2];
#             self.fig.add_shape(
#                 type="line",
#                 x0=x1, y0=y1, x1=x2, y1=y2,
#                 line=dict(
#                     color="#333",
#                     width=4,
#                 ),
#                 layer="below"
#             )
            
#         for i in self.ans:
#             p1,p2 = i;
#             index1=self.label_hash[p1];
#             index2=self.label_hash_2[p2];
#             x1=self.pos_x[index1];y1=self.pos_y[index1];
#             x2=self.pos_x_2[index2];y2=self.pos_y_2[index2];
#             self.fig.add_shape(
#                 type="line",
#                 x0=x1, y0=y1, x1=x2, y1=y2,
#                 line=dict(
#                     color="#6175c1",
#                     width=4,
#                 ),
#                 layer="below"
#             )
#             self.hover_text[index1]="connect {}".format(p2);
#             self.hover_text_2[index2]="connect {}".format(p1);
            
#         pos_x_final = self.pos_x+self.pos_x_2
#         pos_y_final = self.pos_y+self.pos_y_2
#         values_final = self.values + self.values_2
#         labels_final = self.labels + self.labels_2
#         self.fig.add_trace(go.Scatter(x=pos_x_final,
#                         y=pos_y_final,
#                         mode='markers+text',
#                         marker=dict(symbol='circle-dot',
#                                         size=50,
#                                         # color='#5B91D9',  
#                                         color=values_final,
#                                         colorscale="peach",
#                                         showscale=True,
#                                         # line=dict(color='rgb(50,50,50)', width=1)
#                                         ),
#                         text=labels_final,
#                         hoverinfo='text',
#                         hovertext=self.hover_text+self.hover_text_2,
#                         #textposition="top center",
#                         textfont=dict(family='sans serif',
#                         size=18,
#                         color='#000000'
#                             ),
#                         opacity=0.8,
#                         ))
#         # fig.add_trace(go.Scatter(x=pos_x_2,
#         #                   y=pos_y_2,
#         #                   mode='markers+text',
#         #                   name='tree2',
#         #                   marker=dict(symbol='circle-dot',
#         #                                 size=40,
#         #                                 color=values_2,
#         #                                 colorscale="orrd",
#         #                                 showscale=True,
#         #                             ),
#         #                   text=labels_2,
#         #                   hovertext=hover_text_2,
#         #                   hoverinfo="text",
#         #                   textfont=dict(family='sans serif',
#         #                   size=20,
#         #                   color='#000'
#         #                     ),
#         #                   opacity= 0.8,
#         #                   showlegend=False,

#         #                 #   legend=None,
#         #                 #   color=values_2,
#         #                 #   color_continuous_scale="orrd",
#         #                   ))
        
#         self.fig.update_layout(  
#             xaxis= dict(showline=False, # hide axis line, grid, ticklabels and  title
#                     zeroline=False,
#                     showgrid=False,
#                     showticklabels=False,
#                     ),
#             yaxis=dict(showline=False, # hide axis line, grid, ticklabels and  title
#                     zeroline=False,
#                     showgrid=False,
#                     showticklabels=False,
#                     ),
#         width=1000, height=500)
        
#         # self.fig.show()
#     def show_fig(self):
#         self.fig.show();
#     def save_fig(self,path):
#         self.fig.write_image(path)


class show_graph:
    def __init__(self,ans,root1,root2,color=None):
        self.ans = ans;
        
        self.pos_x=[];
        self.pos_y=[];
        self.edges=[];
        self.label_hash = dict(); 
        self.labels = [];
        self.hover_text =[];
        self.values=[];
        self.cnt = 0;
        self.celltypes1=[]
        self.celltypes2=[]

        self.color = color;
        self.pos_x_2=[];
        self.pos_y_2=[];
        self.edges_2=[];
        self.label_hash_2 = dict(); 
        self.labels_2 = []
        self.hover_text_2 = []
        self.values_2=[];
        self.cnt2 = 0;
        self.fig = go.Figure();
        
        self.root1 = root1;
        self.root2 = root2;
        self.height = 5;
        self.run_graph();
        
    def cal_tree_pos(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values,celltype):
        mid = (l+r)/2
        pos_x.append(mid);
        pos_y.append(h)
        label_hash[now.name]=self.cnt;
        num_son = len(now.son);
        labels.append("{}".format(now.name));
        if(len(now.name.split('_')) >1):
            celltype.append(now.name.split('_')[1])
        else:
            celltype.append('empty');
        
        edges.append((f.name,now.name));
        hover_text.append("connect Empty");
        values.append(np.linalg.norm(now.value));
        if(num_son == 0):
            return;
        length = (r-l)/num_son;
        for i in range(num_son):
            self.cnt+=1;
            self.cal_tree_pos(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values,celltype);
            
    def cal_tree_pos2(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values,celltype):
        mid = (l+r)/2
        pos_x.append(mid);
        pos_y.append(h)
        label_hash[now.name]=self.cnt2;
        num_son = len(now.son);
        labels.append("{}".format(now.name));
        if(len(now.name.split('_')) >1):
            celltype.append(now.name.split('_')[1])
        else:
            celltype.append('empty');
        edges.append((f.name,now.name));
        hover_text.append("connect Empty");
        values.append(np.linalg.norm(now.value))
        if(num_son == 0):
            return;

        length = (r-l)/num_son;
        for i in range(num_son):
            self.cnt2+=1;
            self.cal_tree_pos2(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values,celltype);
            
    def run_graph(self):
        
        self.cal_tree_pos(self.root1,1,20,self.height,self.root1, self.pos_x,self.pos_y,self.edges,self.label_hash,self.labels,self.hover_text,self.values,self.celltypes1);
        
        self.cal_tree_pos2(self.root2,18,37,self.height-1.5,self.root2, self.pos_x_2,self.pos_y_2,self.edges_2,self.label_hash_2,self.labels_2,self.hover_text_2,self.values_2,self.celltypes2);
        unique_categories = list(set(self.celltypes1+self.celltypes2))
        if(self.color == None):
            category_colors = {cat: f'#{random.randint(0, 0xFFFFFF):06x}' for cat in unique_categories}
        elif(len(self.color)>= len(unique_categories)):
            category_colors = {cat: col  for cat,col in zip(unique_categories,self.color[:len(unique_categories)])}
        else:
            category_colors = {cat: f'#{random.randint(0, 0xFFFFFF):06x}' for cat in unique_categories}
            
        category_colors['empty'] = '#fff'        
        print(category_colors)
        
        for i in self.edges:
            p1,p2 = i;
            index1=self.label_hash[p1];
            index2=self.label_hash[p2];

            x1=self.pos_x[index1];y1=self.pos_y[index1];
            x2=self.pos_x[index2];y2=self.pos_y[index2];
            self.fig.add_shape(
                type="line",
                x0=x1, y0=y1, x1=x2, y1=y2,
                line=dict(
                    color="#333",
                    width=4,
                ),
                layer="below"
            )

        for i in self.edges_2:
            p1,p2 = i;

            index1=self.label_hash_2[p1];
            index2=self.label_hash_2[p2];

            x1=self.pos_x_2[index1];y1=self.pos_y_2[index1];
            x2=self.pos_x_2[index2];y2=self.pos_y_2[index2];
            self.fig.add_shape(
                type="line",
                x0=x1, y0=y1, x1=x2, y1=y2,
                line=dict(
                    color="#333",
                    width=4,
                ),
                layer="below"
            )
            
        for i in self.ans:
            p1,p2 = i;
            index1=self.label_hash[p1];
            index2=self.label_hash_2[p2];
            x1=self.pos_x[index1];y1=self.pos_y[index1];
            x2=self.pos_x_2[index2];y2=self.pos_y_2[index2];
            self.fig.add_shape(
                type="line",
                x0=x1, y0=y1, x1=x2, y1=y2,
                line=dict(
                    color="#6175c1",
                    width=4,
                ),
                layer="below"
            )
            self.hover_text[index1]="connect {}".format(p2);
            self.hover_text_2[index2]="connect {}".format(p1);
          
        for cell_type in unique_categories:
            indexs = [i for i in range(len(self.celltypes1)) if self.celltypes1[i] == cell_type]
            x_cell1 = [self.pos_x[i] for i in indexs]
            y_cell1 = [self.pos_y[i] for i in indexs]
            label1 = [self.hover_text[i] for i in indexs]
            name1 = [self.labels[i] for i in indexs]
            indexs = [i for i in range(len(self.celltypes2)) if self.celltypes2[i] == cell_type]
            x_cell2 = [self.pos_x_2[i] for i in indexs]
            y_cell2 = [self.pos_y_2[i] for i in indexs]
            label2 = [self.hover_text_2[i] for i in indexs]
            name2 = [self.labels_2[i] for i in indexs]
            x_cell = x_cell1+x_cell2
            y_cell = y_cell1 + y_cell2
            label = label1+label2
            # Add scatter trace for the current cell type
            self.fig.add_trace(go.Scatter(
                x=x_cell,
                y=y_cell,
                mode='markers+text',
                name=cell_type,
                legendgroup=cell_type,
                marker=dict(
                    symbol='circle-dot',
                    size=35,
                    color=category_colors[cell_type]  # Set color based on cell type
                ),
                hoverinfo='text',
                hovertext=label,
                text = name1+name2,
                textfont=dict(family='sans serif',
                size=15,
                color='#000000'
                    ),
            ))
        # pos_x_final = self.pos_x+self.pos_x_2
        # pos_y_final = self.pos_y+self.pos_y_2
        # values_final = self.values + self.values_2
        # labels_final = self.labels + self.labels_2
        # self.fig.add_trace(go.Scatter(x=pos_x_final,
        #                 y=pos_y_final,
        #                 mode='markers+text',
        #                 marker=dict(symbol='circle-dot',
        #                                 size=50,
        #                                 color=values_final,
        #                                 colorscale="peach",
        #                                 ),
        #                 text=labels_final,
        #                 hoverinfo='text',
        #                 hovertext=self.hover_text+self.hover_text_2,
        #                 textfont=dict(family='sans serif',
        #                 size=18,
        #                 color='#000000'
        #                     ),
        #                 opacity=0.8,
        #                 ))
        
        self.fig.update_layout(  
            xaxis= dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    ),
            yaxis=dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    ),
        width=1000, height=500)
        
        # self.fig.show()
    def show_fig(self):
        self.fig.show();
    def save_fig(self,path):
        self.fig.write_image(path)


class show_tree:
    def __init__(self,root1,color=None):
        
        self.pos_x=[];
        self.pos_y=[];
        self.edges=[];
        self.label_hash = dict(); 
        self.labels = [];
        self.hover_text =[];
        self.values=[];
        self.cnt = 0;
        self.height=5;
        self.celltypes = [];
        self.fig = go.Figure();
        self.color = color;
        self.root1 = root1;
        self.run_graph();
        
    def cal_tree_pos(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values,celltypes):
        mid = (l+r)/2
        pos_x.append(mid);
        pos_y.append(h)
        label_hash[now.name]=self.cnt;
        num_son = len(now.son);
        labels.append("{}".format(now.name));
        if(len(now.name.split('_')) >1):
            celltypes.append(now.name.split('_')[1])
        else:
            celltypes.append('empty');
        edges.append((f.name,now.name));
        hover_text.append("connect Empty");
        values.append(np.linalg.norm(now.value));
        if(num_son == 0):
            return;
        length = (r-l)/num_son;
        for i in range(num_son):
            self.cnt+=1;
            self.cal_tree_pos(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values,celltypes);
            

    def run_graph(self):
        
        self.cal_tree_pos(self.root1,1,10,self.height,self.root1, self.pos_x,self.pos_y,self.edges,self.label_hash,self.labels,self.hover_text,self.values,self.celltypes);
        
        unique_categories = list(set(self.celltypes))
        if(self.color == None):
            category_colors = {cat: f'#{random.randint(0, 0xFFFFFF):06x}' for cat in unique_categories}
        elif(len(self.color)>= len(unique_categories)):
            category_colors = {cat: col  for cat,col in zip(unique_categories,self.color[:len(unique_categories)])}
        else:
            category_colors = {cat: f'#{random.randint(0, 0xFFFFFF):06x}' for cat in unique_categories}
            
        category_colors['empty'] = '#fff'
        print(category_colors)
        for i in self.edges:
            p1,p2 = i;
            index1=self.label_hash[p1];
            index2=self.label_hash[p2];

            x1=self.pos_x[index1];y1=self.pos_y[index1];
            x2=self.pos_x[index2];y2=self.pos_y[index2];
            self.fig.add_shape(
                type="line",
                x0=x1, y0=y1, x1=x2, y1=y2,
                line=dict(
                    color="#333",
                    width=2,
                ),
                layer="below"
            )
        # Iterate over unique cell types
        for cell_type in unique_categories:
            indexs = [i for i in range(len(self.celltypes)) if self.celltypes[i] == cell_type]
            x_cell = [self.pos_x[i] for i in indexs]
            y_cell = [self.pos_y[i] for i in indexs]
            
            # Add scatter trace for the current cell type
            self.fig.add_trace(go.Scatter(
                x=x_cell,
                y=y_cell,
                mode='markers+text',
                name=cell_type,
                legendgroup=cell_type,
                marker=dict(
                    symbol='circle-dot',
                    size=10,
                    color=category_colors[cell_type]  # Set color based on cell type
                ),
                # text=[self.labels[i] for i in indexs],
                hoverinfo='text',
                hovertext=[self.labels[i] for i in indexs],
                # #textposition="top center",
                # textfont=dict(family='sans serif',
                # size=18,
                # color='#000000'
                # ),
                # opacity=0.8,
            ))

        # self.fig.add_trace(go.Scatter(x=self.pos_x,
        #                 y=self.pos_y,
        #                 mode='markers+text',
        #                 marker=dict(symbol='circle-dot',
        #                                 size=50,
        #                                 # color='#5B91D9',  
        #                                 # color=self.values,
        #                                 color=[category_colors[cat] for cat in self.celltypes],
        #                                 # colorscale="peach",
        #                                 # showscale=True,
        #                                 # line=dict(color='rgb(50,50,50)', width=1)
        #                                 ),
        #                 text=self.labels,
        #                 hoverinfo='text',
        #                 # hovertext=self.hover_text,
        #                 #textposition="top center",
        #                 textfont=dict(family='sans serif',
        #                 size=18,
        #                 color='#000000'
        #                     ),
        #                 opacity=0.8,
        #                 ))
        self.fig.update_layout(  
            xaxis= dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    ),
            yaxis=dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    ),
        width=1000, height=500
        )
        
        # self.fig.show()
    def show_fig(self):
        self.fig.show();
              
              
               
def show_the_tree(folder_path1,after=False):
    nodes1 = build_hyper_tree_from_folder(folder_path1,after)
    show_tree(nodes1[0]).show_fig()

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
        now.subson.extend(i.subson)

    path.remove(now)

def build_hyper_tree_from_folder(folder_path,after=False,mst1=False):
    """
    Build the tree from the folder
    """
    if(mst1):
        pos_1 = pd.read_csv(folder_path + 'datas.csv')
        pos = pos_1.set_index(pos_1.columns[0]).values
        edge = np.load(folder_path + "datalink.npy");
        father_name = np.load(folder_path + "dataname.npy")
        father_name = father_name.astype(np.int)
        xys = np.load(folder_path+'dataxy.npy');
        n_points = len(pos);
        nodes = [node(name=str(i),son=[]) for i in range(n_points)];
        for i in range(n_points-1):
            if(edge[i]!=-1):
                nodes[edge[i]].son.append(nodes[i])
            nodes[i].name = str(father_name[i])
            nodes[i].value = pos[father_name[i]]
        return nodes
    
    pos_1 = pd.read_csv(folder_path + 'datas.csv')
    pos = pos_1.set_index(pos_1.columns[0]).values
    
    if(after):
        edge = np.load(folder_path + "datalink_merge.npy");
        father_name = np.load(folder_path + "dataname_merge.npy")
        father_name = father_name.astype(np.int)
        xys = np.load(folder_path+'dataxy_merge.npy');    
    else:
        edge = np.load(folder_path + "datalink.npy");
        father_name = np.load(folder_path + "dataname.npy")
        father_name = father_name.astype(np.int)
        xys = np.load(folder_path+'dataxy.npy');
        
    n = len(edge)
    n_points = len(pos);
    nodes = [node(name=str(i),son=[]) for i in range(n)];
    for i in range(n):
        if(edge[i]!=-1):
            nodes[edge[i]].son.append(nodes[i])
        nodes[i].name = str(father_name[i])
        if(father_name[i]<n_points):
            nodes[i].value = pos[father_name[i]]
        else:
            nodes[i].value = 0.0
        nodes[i].hyper = xys[i];
            
    def test(now):
        for i in now.son:
            test(i);
        if(now.son!=[]):
            son_values = []
            all = 0;
            for i in now.son:
                values = hyp_dist(torch.tensor(now.hyper),torch.tensor(i.hyper))
                values = values.numpy()
                all += values;
                son_values.append(values*i.value)
            son_values = np.array(son_values)
            son_values /=all
            now.value = son_values.sum(axis=0)
    test(nodes[0]);  
    for i in nodes:
        if(int(i)<n_points):
            i.subson = [int(i)]
        else:
            i.subson=[]
    deep_search_tree(nodes[0],0,[],nodes[0])
    return nodes

def search_tree(now,c,c1,n):
    """
    Merge the tree nodes according of the c
    """
    if(len(now.son) < 2):
        return now,True;
    
    sons = []
    flag = True
    for i in range(len(now.son)):
        son,f1 = search_tree(now.son[i],c,c1,n);
        now.son[i]=son
        sons.append(son)
        flag = (flag and f1)
    maxn = 10000;
    maxnn = None;
    
    for i in sons:
        if(int(i)<n):
            dis = hyp_dist(i.hyper,now.hyper)
            if(dis<=c and dis<maxn):
                maxn = dis;
                maxnn = i;
    lson = now.son[0]
    rson = now.son[1]
    if(flag and maxnn != None and int(now)>=n and hyp_dist(lson.hyper,rson.hyper)<=c1):
        sons.remove(maxnn);
        print(now,maxnn)
        now = maxnn;
        now.son= now.son + sons;
        flag = False;
    
    # lson = search_tree(now.son[0],c,n);
    # now.son[0] = lson;
    # rson = search_tree(now.son[1],c,n);
    # now.son[1] = rson
    # distance_l = hyp_dist(lson.hyper,now.hyper);
    # distance_r = hyp_dist(rson.hyper,now.hyper)
    
    # if(distance_l< distance_r and int(lson)<n):
    #     if(distance_l<=c):
    #         now = lson.copy();
    #         now.son.append(rson);
    #         pass;
    # elif(int(rson)<n):
    #     if(distance_r<=c):
    #         now = rson.copy();
    #         now.son.append(lson)
    #         pass

    return now,flag;

def find_path_root(now,dfs,path,dfs_node,f):
    """
    Find the path to the root
    """
    now.path=path.copy();
    now.f=f
    now.dfs=dfs;
    path.append(now);
    dfs_node.append(now);
    for i in now.son:
        dfs=find_path_root(i,dfs+1,path,dfs_node,now);
        
    path.remove(now)
    now.num_son = dfs-now.dfs;
    return dfs

def find_indegree(lists,indegree):
    """
    Find the indegrees
    """
    ans=[]
    for i in lists:
        if(i.indegree == indegree):
            ans.append(i);
    return ans;

def run_alignment_linear(nodes1,nodes2):
    """
    Alignment two trees by linear programming
    """
    values1 = np.array([i.value for i in nodes1])
    values2 = np.array([i.value for i in nodes2])
    similarities =np.zeros((len(values1),len(values2)))
    for i in range(len(values1)):
        for j in range(len(values2)):
            similarities[i][j]=np.corrcoef(values1[i],values2[j])[0][1]
            
    n = len(nodes1)
    m = len(nodes2)
    set_I = range(0, n)
    set_J = range(0, m)
    c = {(i,j): similarities[i][j] for i in set_I for j in set_J}
    dfs_node1=[]
    dfs_node2=[]
    root1 = nodes1[0]
    root2 = nodes2[0]
    find_path_root(root1,0,[],dfs_node1,root1)
    find_path_root(root2,0,[],dfs_node2,root2)
    x_vars  = {(i,j):plp.LpVariable(cat=plp.LpBinary, name="x_{0}_{1}".format(i,j)) for i in set_I for j in set_J}
    x_vars[0,0] =plp.LpVariable(cat='Integer',lowBound=1,upBound=1, name="x_0_0")
    opt_model = plp.LpProblem(name="MIP Model")
    for i in set_I:
        opt_model.addConstraint(
        plp.LpConstraint(e=plp.lpSum(x_vars[i,j] for j in set_J),
                        sense=plp.LpConstraintGE,
                        rhs=1,
                        name="constraintI{0}".format(i)))

    # for j in set_J:
    #     opt_model.addConstraint(
    #     plp.LpConstraint(e=plp.lpSum(x_vars[i,j] for i in set_I),
    #                     sense=plp.LpConstraintGE,
    #                     rhs=1,
    #                     name="constraintJ{0}".format(j))) 
    for i in dfs_node1:
        for j in dfs_node2:
            for k in i.path:
                for l in j.path:
                    if(k==[]or l==[]):
                        continue;
                    # print(i,j,k,l)
                    opt_model.addConstraint(
                    plp.LpConstraint(e=x_vars[i.dfs,j.dfs]+x_vars[i.dfs,l.dfs]+x_vars[k.dfs,j.dfs],
                                    sense=plp.LpConstraintLE,
                                    rhs=2,
                                    name="constraint{}_{}_{}_{}_1".format(i,j,k,l)))
    for i in dfs_node1:
        if(len(i.son)==2):
            l=i.son[0];
            r=i.son[1];
            for j in dfs_node2:
                opt_model.addConstraint(
                plp.LpConstraint(e=x_vars[l.dfs,j.dfs]+x_vars[r.dfs,j.dfs]-x_vars[i.dfs,j.dfs],
                                    sense=plp.LpConstraintLE,
                                    rhs=1,
                                    name="constraint{}_{}_{}_{}_2".format(i,j,l,r)))
    for j in dfs_node2:
        if(len(j.son)==2):
            l=j.son[0];
            r=j.son[1];
            for i in dfs_node1:
                opt_model.addConstraint(
                plp.LpConstraint(e=x_vars[i.dfs,l.dfs]+x_vars[i.dfs,r.dfs]-x_vars[i.dfs,j.dfs],
                                    sense=plp.LpConstraintLE,
                                    rhs=1,
                                    name="constraint{}_{}_{}_{}_3".format(i,j,l,r)))

    objective = plp.lpSum(x_vars[i,j] * (1-c[i,j]) for i in set_I for j in set_J)
    opt_model.sense = plp.LpMinimize
    opt_model.setObjective(objective)
    opt_model.solve()
    # print('SOLUTION:')
    # for v in opt_model.variables():
    #     print(f'\t\t{v.name} = {v.varValue}')

    # print('\n') # Prints a blank line
    print(f'OBJECTIVE VALUE: {opt_model.objective.value()}')
    
    result_node = []
    anslist = []
    for v in opt_model.variables():
        if(v.value()==0):
            continue
        l = int(v.name.split('_')[1])
        r = int(v.name.split('_')[2])
        tn = newnode(dfs_node1[l],dfs_node2[r])
        result_node.append(tn);
        anslist.append((str(dfs_node1[l]),str(dfs_node2[r])))
        
    for i in range(len(result_node)):
        for j in range(len(result_node)):
            if(i==j):
                continue
            p1 = result_node[i]
            p2 = result_node[j]
            if((p1.node1 in p2.node1.path or p1.node1 == p2.node1) and (p1.node2 in p2.node2.path or p1.node2==p2.node2)):
                result_node[i].edge.append(result_node[j])
                result_node[j].indegree +=1
                
    for i in range(len(result_node)):
        for j in range(len(result_node)):
            if(i==j):
                continue
            p1 = result_node[i]
            p2 = result_node[j]
    root=find_indegree(result_node,0)[0]
    
    c=0;z=0

    for i in result_node:
        ans = str(i).split('_')
        if(len(ans) == 4):
            c+=1
            if(ans[1] == ans[3]):
                z+=1
                
    if(c!=0):
        print('cell type rate:{}'.format(z/c))
        return z/c,result_node,anslist;
    else:
        return 0,result_node,anslist;
    

def run_alignment(nodes1,nodes2,folder_path1,folder_path2,meta_list1,meta_list2):
    """
    Alignment two trees by dynmaic programming
    """
    T=tree_alignment(nodes1[0],nodes2[0],1);
    minn = T.run_alignment();
    T.show_ans();
    ans = T.get_ans()
    G=show_graph(ans,nodes1[0],nodes2[0]);
    # G.show_fig()
    G.save_fig(folder_path1+'alignment.png')
    G.save_fig(folder_path2+'alignment.png')
    
    n1 =len(nodes1)
    n2 =len(nodes2)

    print("average cost for one node:{}\n".format(minn/(n1+n2)))
    
    c=0;z=0
    anslist = [];
    for i,j in ans:
        i=int(i.split('_')[0])
        j=int(j.split('_')[0])
        if(i<len(meta_list1) and j <len(meta_list2)):
            c+=1
            if(meta_list1[i]==meta_list2[j]):
                z+=1;
            anslist.append((i,j))
    if(c!=0):
        print('cell type rate:{}'.format(z/c))
        return z/c,anslist,ans;
    else:
        return 0,anslist,ans;