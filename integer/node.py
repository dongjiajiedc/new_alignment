
import numpy as np
import pandas as pd


class node:
    def __init__(self,name,son):
        self.name=name;
        self.son=son;
        self.f=None;
        self.path=None;
        self.num_son=0;
        self.dfs=None;
    def __repr__(self):
            return self.name
    def __str__(self):
        return self.name
    def __int__(self):
        return int(self.name);
class newnode:
    def __init__(self,node1,node2):
        self.node1 = node1
        self.node2 = node2
        self.f = None
        self.edge = [];
        self.indegree = 0;
    def __str__(self):
        return "{}_{}".format(self.node1,self.node2)
    def __repr__(self):
        return "{}_{}".format(self.node1,self.node2)
