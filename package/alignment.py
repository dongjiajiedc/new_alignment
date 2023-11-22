
import numpy as np
import pandas as pd
import math
import plotly.graph_objs as go
import os
import csv
import itertools


class node:
    def __init__(self,value=None,son=[],name=''):
        self.value = value;
        self.son = son;
        self.name =name;
        self.f = None;
        self.depth=0;
        self.subson= [];
    def __repr__(self):
        return self.name
    def __str__(self):
        return self.name
    def copy(self):
        return node(self.value,self.son,self.name)
    def rest(self,n):
        all = [i for i in range(n)];
        result = [element for element in all if element not in self.subson];
        return result

        
    # def __lt__(self, other):
    #     return self.depth < other.depth
    
class tree_alignment:
    def __init__(self,root1,root2,cost1):
        self.cost1 = cost1;
        self.dp = dict();
        self.forestdp = dict();
        self.anslist = [];
        self.ansnodes = [];
        self.root1 = root1;
        self.root2 = root2;
        self.minn = math.inf;
        self.cal_depth(root1,0)
        self.cal_depth(root2,0)
        pass;

    def cal_depth(self, now,d):
        now.depth = d;
        for i in now.son:
            self.cal_depth(i,d+1);
            
    def cost(self,i,j):
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
            df = pd.DataFrame(
                {"A": i.value, "B":j.value})
            mincost = 1 - df.corr(method="spearman").iloc[0, 1]
            return mincost
            # return np.linalg.norm(i.value-j.value);
    
    def cal_tree(self,i,j):
        
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
class show_graph:
    def __init__(self,ans,root1,root2):
        self.ans = ans;
        
        self.pos_x=[];
        self.pos_y=[];
        self.edges=[];
        self.label_hash = dict(); 
        self.labels = [];
        self.hover_text =[];
        self.values=[];
        self.cnt = 0;
        
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
        
    def cal_tree_pos(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values):
        mid = (l+r)/2
        pos_x.append(mid);
        pos_y.append(h)
        label_hash[now.name]=self.cnt;
        num_son = len(now.son);
        labels.append("{}".format(now.name));
        edges.append((f.name,now.name));
        hover_text.append("connect Empty");
        values.append(np.linalg.norm(now.value));
        if(num_son == 0):
            return;
        length = (r-l)/num_son;
        for i in range(num_son):
            self.cnt+=1;
            self.cal_tree_pos(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values);
            
    def cal_tree_pos2(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values):
        mid = (l+r)/2
        pos_x.append(mid);
        pos_y.append(h)
        label_hash[now.name]=self.cnt2;
        num_son = len(now.son);
        labels.append("{}".format(now.name));
        edges.append((f.name,now.name));
        hover_text.append("connect Empty");
        values.append(np.linalg.norm(now.value))
        if(num_son == 0):
            return;

        length = (r-l)/num_son;
        for i in range(num_son):
            self.cnt2+=1;
            self.cal_tree_pos2(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values);
            
    def run_graph(self):
        
        self.cal_tree_pos(self.root1,1,20,self.height,self.root1, self.pos_x,self.pos_y,self.edges,self.label_hash,self.labels,self.hover_text,self.values);
        
        self.cal_tree_pos2(self.root2,18,37,self.height-1.5,self.root2, self.pos_x_2,self.pos_y_2,self.edges_2,self.label_hash_2,self.labels_2,self.hover_text_2,self.values_2);
        
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
            
        pos_x_final = self.pos_x+self.pos_x_2
        pos_y_final = self.pos_y+self.pos_y_2
        values_final = self.values + self.values_2
        labels_final = self.labels + self.labels_2
        self.fig.add_trace(go.Scatter(x=pos_x_final,
                        y=pos_y_final,
                        mode='markers+text',
                        marker=dict(symbol='circle-dot',
                                        size=50,
                                        # color='#5B91D9',  
                                        color=values_final,
                                        colorscale="peach",
                                        showscale=True,
                                        # line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                        text=labels_final,
                        hoverinfo='text',
                        hovertext=self.hover_text+self.hover_text_2,
                        #textposition="top center",
                        textfont=dict(family='sans serif',
                        size=18,
                        color='#000000'
                            ),
                        opacity=0.8,
                        ))
        # fig.add_trace(go.Scatter(x=pos_x_2,
        #                   y=pos_y_2,
        #                   mode='markers+text',
        #                   name='tree2',
        #                   marker=dict(symbol='circle-dot',
        #                                 size=40,
        #                                 color=values_2,
        #                                 colorscale="orrd",
        #                                 showscale=True,
        #                             ),
        #                   text=labels_2,
        #                   hovertext=hover_text_2,
        #                   hoverinfo="text",
        #                   textfont=dict(family='sans serif',
        #                   size=20,
        #                   color='#000'
        #                     ),
        #                   opacity= 0.8,
        #                   showlegend=False,

        #                 #   legend=None,
        #                 #   color=values_2,
        #                 #   color_continuous_scale="orrd",
        #                   ))
        
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
    def __init__(self,root1):
        
        self.pos_x=[];
        self.pos_y=[];
        self.edges=[];
        self.label_hash = dict(); 
        self.labels = [];
        self.hover_text =[];
        self.values=[];
        self.cnt = 0;
        self.height=5;
        self.fig = go.Figure();
        
        self.root1 = root1;
        self.run_graph();
        
    def cal_tree_pos(self,now,l,r,h,f,pos_x,pos_y,edges,label_hash,labels,hover_text,values):
        mid = (l+r)/2
        pos_x.append(mid);
        pos_y.append(h)
        label_hash[now.name]=self.cnt;
        num_son = len(now.son);
        labels.append("{}".format(now.name));
        edges.append((f.name,now.name));
        hover_text.append("connect Empty");
        values.append(np.linalg.norm(now.value));
        if(num_son == 0):
            return;
        length = (r-l)/num_son;
        for i in range(num_son):
            self.cnt+=1;
            self.cal_tree_pos(now.son[i],l+i*length,l+(i+1)*length,h-1.5,now,pos_x,pos_y,edges,label_hash,labels,hover_text,values);
            

    def run_graph(self):
        
        self.cal_tree_pos(self.root1,1,10,self.height,self.root1, self.pos_x,self.pos_y,self.edges,self.label_hash,self.labels,self.hover_text,self.values);
        
        
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
        self.fig.add_trace(go.Scatter(x=self.pos_x,
                        y=self.pos_y,
                        mode='markers+text',
                        marker=dict(symbol='circle-dot',
                                        size=50,
                                        # color='#5B91D9',  
                                        color=self.values,
                                        colorscale="peach",
                                        showscale=True,
                                        # line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                        text=self.labels,
                        hoverinfo='text',
                        # hovertext=self.hover_text,
                        #textposition="top center",
                        textfont=dict(family='sans serif',
                        size=18,
                        color='#000000'
                            ),
                        opacity=0.8,
                        ))
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
        


        
def run_alignment(folder_path1,folder_path2):

    pos_1 = pd.read_csv(folder_path1+"datas.csv",index_col="Unnamed: 0").sort_index().values
    pos_2 = pd.read_csv(folder_path2+"datas.csv",index_col="Unnamed: 0").sort_index().values
    edge_1 = np.load(folder_path1+"datalink.npy");
    edge_2 = np.load(folder_path2+"datalink.npy");
    mer1 = np.load(folder_path1+"datamerge.npy");
    mer2 = np.load(folder_path2+"datamerge.npy");

    n1 = len(pos_1)
    n2 = len(pos_2)

    root1 = -1;
    for i,j in edge_1:
        root1 = max(root1,i);
    length1 = root1 + 1;

    root2 = -1;
    for i,j in edge_2:
        root2 = max(root2,i);
    length2 = root2 + 1;

    nodes1 = [node(name=str(i),son=[]) for i in range(length1)]
    nodes2 = [node(name=str(i),son=[]) for i in range(length2)]
    for i,j in edge_1:
        nodes1[i].son.append(nodes1[j])
    for i,j in edge_2:
        nodes2[i].son.append(nodes2[j])
    for i in range(len(pos_1)):
        nodes1[i].value = pos_1[i];
        nodes1[i].subson = [nodes1[i].name];
    for i in range(len(pos_2)):
        nodes2[i].value = pos_2[i];
        nodes2[i].subson = [nodes2[i].name]

        
    for i in range(len(pos_1),length1):
        if(mer1[i]!= -1):
            nodes1[i].value = nodes1[mer1[i]].value
            nodes1[i].name = nodes1[mer1[i]].name+'(t)'
            nodes1[i].subson  = nodes1[mer1[i]].subson;
        else:
            for son in nodes1[i].son:
                nodes1[i].subson.extend(son.subson);
        
    for i in range(len(pos_2),length2):
        if(mer2[i]!= -1):
            nodes2[i].value = nodes2[mer2[i]].value
            nodes2[i].name = nodes2[mer2[i]].name+'(t)'
            nodes2[i].subson = nodes2[mer2[i]].subson
        else:
            for son in nodes2[i].son:
                nodes2[i].subson.extend(son.subson);

                
    adata1 = pd.read_csv(folder_path1+"data_cell.csv")
    type1 = pd.read_csv(folder_path1+"data_type.csv")[['Unnamed: 0','leiden']]
    datas1 = type1.merge(adata1,how="inner",on="Unnamed: 0")
    datas1 = datas1.set_index("Unnamed: 0")
            
    adata2 = pd.read_csv(folder_path2+"data_cell.csv")
    type2 = pd.read_csv(folder_path2+"data_type.csv")[['Unnamed: 0','leiden']]
    datas2 = type2.merge(adata2,how="inner",on="Unnamed: 0")
    datas2 = datas2.set_index("Unnamed: 0")
            
    for i in range(len(pos_1),length1):
        if(mer1[i]!=-1):
            continue;
        med = []
        for sub in nodes1[i].subson:
            subdata = datas1[datas1['leiden']==int(sub)];
            med.extend(subdata.values[:,1:].tolist());
        med = np.array(med);
        p = pd.DataFrame(med);
        nodes1[i].value = p.mean(axis=0).values;
        
    for i in range(len(pos_2),length2):

        if(mer2[i]!=-1):
            continue;
        med = []
        for sub in nodes2[i].subson:
            subdata = datas2[datas2['leiden']==int(sub)];
            med.extend(subdata.values[:,1:].tolist());
        med = np.array(med);
        p = pd.DataFrame(med);
        nodes2[i].value = p.mean(axis=0).values;

    # for i in range(len(pos_1),length1):
    #     sum = np.array([0 for j in range(len(pos_1[0]))],dtype=np.float32);
    #     med = [];
    #     for son in nodes1[i].son:
    #         sum = sum + son.value;
    #         med.append(son.value);
    #     sum= sum / max(len(nodes1[i].son),1);
    #     med = np.array(med);
    #     p = pd.DataFrame(med);
    #     nodes1[i].value = p.mean(axis=0).values;
    #     nodes1[i].value = sum;
        

        
    # for i in range(len(pos_2),len(pos_2)*2-1):
    #     sum = np.array([0 for j in range(len(pos_2[0]))],dtype=np.float32);
    #     med = [];
    #     for son in nodes2[i].son:
    #         sum= sum + son.value;
    #         med.append(son.value);
    #     sum= sum / max(len(nodes2[i].son),1);
    #     med = np.array(med);
    #     p =pd.DataFrame(med);
    #     nodes2[i].value = p.mean(axis=0).values;
    #     nodes2[i].value = sum
    T=tree_alignment(nodes1[root1],nodes2[root2],1);
    minn = T.run_alignment();
    T.show_ans();
    ans = T.get_ans()
    ans
    G=show_graph(ans,nodes1[root1],nodes2[root2]);
    G.show_fig()
def show_the_tree(folder_path1):
    pos_1 = pd.read_csv(folder_path1+"datas.csv",index_col="Unnamed: 0").sort_index().values
    edge_1 = np.load(folder_path1+"datalink.npy");
    mer1 = np.load(folder_path1+"datamerge.npy");

    n1 = len(pos_1)

    root1 = -1;
    for i,j in edge_1:
        root1 = max(root1,i);
    length1 = root1 + 1;    
    nodes1 = [node(name=str(i),son=[]) for i in range(length1)]
    for i,j in edge_1:
        nodes1[i].son.append(nodes1[j])

    for i in range(len(pos_1)):
        nodes1[i].value = pos_1[i];
        nodes1[i].subson = [nodes1[i].name];

        
    for i in range(len(pos_1),length1):
        if(mer1[i]!= -1):
            nodes1[i].value = nodes1[mer1[i]].value
            nodes1[i].name = nodes1[mer1[i]].name+'(t)'
            nodes1[i].subson  = nodes1[mer1[i]].subson;
        else:
            for son in nodes1[i].son:
                nodes1[i].subson.extend(son.subson);
    adata1 = pd.read_csv(folder_path1+"data_cell.csv")
    type1 = pd.read_csv(folder_path1+"data_type.csv")[['Unnamed: 0','leiden']]
    datas1 = type1.merge(adata1,how="inner",on="Unnamed: 0")
    datas1 = datas1.set_index("Unnamed: 0")
    for i in range(len(pos_1),length1):
        if(mer1[i]!=-1):
            continue;
        med = []
        for sub in nodes1[i].subson:
            subdata = datas1[datas1['leiden']==int(sub)];
            med.extend(subdata.values[:,1:].tolist());
        med = np.array(med);
        p = pd.DataFrame(med);
        nodes1[i].value = p.mean(axis=0).values;
    show_tree(nodes1[root1]).show_fig()