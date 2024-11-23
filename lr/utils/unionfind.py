
class  UnionFind:
    
    def __init__(self, n):
        self.n = n
        # self.c = c
        self.parent = [i for i in range(n)]
        self.rank = [0 for i in range(n)]
        self._next_id = n
        self._tree = [-1 for i in range(2*n-1)]
        self._id = [i for i in range(n)]

    def _find(self, i):
        if self.parent[i] == i:
            return i
        else:
            self.parent[i] = self._find(self.parent[i])
            return self.parent[i]

    def find(self, i):
        if (i < 0) or (i > self.n):
            raise ValueError("Out of bounds index.")
        return self._find(i)

    def union(self,  i,  j):
        root_i = self._find(i)
        root_j = self._find(j)
        if root_i == root_j:
            return False
        else:
            
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self._build(root_j, root_i)
                
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self._build(root_i, root_j)
                
            else:
                
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                self._build(root_i, root_j)
                
            return True


    def merge(self,ij):
        for k in range(ij.shape[0]):
            a=ij[k,0];
            b=ij[k,1];
            self.union(a, b)

    def  _build(self, i, j):
        self._tree[self._id[i]] = self._next_id
        self._tree[self._id[j]] = self._next_id
        self._id[i] = self._next_id
        self._next_id += 1

    def sets(self):
        return 2 * self.n - self._next_id
                
        
    def parent(self):
        return [self.parent[i] for i in range(self.n)]

    def tree(self):
        return [self._tree[i] for i in range(len(self._tree))]
    
    
# class  UnionFind:
    
#     def __init__(self, n , pos):
#         self.n = n
#         self.pos = pos
#         # self.c = c
#         self.parent = [i for i in range(n)]
#         self.rank = [0 for i in range(n)]
#         self.vis = [0 for i in range(2*n-1)]
#         self.vis2 = [0 for i in range(2*n-1)]
#         self.mer = [-1 for i in range(2*n-1)]
#         self._next_id = n
#         self._tree = [-1 for i in range(2*n-1)]
#         self._id = [i for i in range(n)]

#     def _find(self, i):
#         if self.parent[i] == i:
#             return i
#         else:
#             self.parent[i] = self._find(self.parent[i])
#             return self.parent[i]

#     def find(self, i):
#         if (i < 0) or (i > self.n):
#             raise ValueError("Out of bounds index.")
#         return self._find(i)

#     def union(self,  i,  j, k=True):
#         root_i = self._find(i)
#         root_j = self._find(j)
#         if root_i == root_j:
#             return False
#         else:
            
#             if self.rank[root_i] < self.rank[root_j]:
#                 self.parent[root_i] = root_j
#                 if(k):
#                     self._build(root_j, root_i)
                
#             elif self.rank[root_i] > self.rank[root_j]:
#                 self.parent[root_j] = root_i
#                 if(k):
#                     self._build(root_i, root_j)
                
#             else:
                
#                 self.parent[root_j] = root_i
#                 self.rank[root_i] += 1
#                 if(k):
#                     self._build(root_i, root_j)
                
#             return True


#     def merge(self,ij):
#         for k in range(ij.shape[0]):
#             a=ij[k,0];
#             b=ij[k,1];
#             if(self.mer[a]!=-1):
#                 a=self.mer[a];
#             if(self.mer[b]!=-1):
#                 b=self.mer[b];
#             self.union(a, b)

#     def  _build(self, i, j):
#         self.vis2[i]=1;
#         self.vis2[j]=1;
        
#         t=np.array(self.pos).tolist()
#         new = np.array(hyp_lca(self.pos[self._id[i]],self.pos[self._id[j]])).tolist()
#         t.append(new);
#         self.pos=torch.tensor(t)
        
#         self._tree[self._id[i]] = self._next_id
#         self._tree[self._id[j]] = self._next_id
#         self._id[i] = self._next_id
#         self._next_id += 1

#     def search(self,k):

#         for i in range(len(self._tree)):
#             if(self.vis[i]):
#                 continue;            
#             if(self._tree[i]==k):
#                 self._tree[i]=self._next_id;  
                
#         for i in range(len(self._id)):
#             if(self.vis[i]):
#                 continue;
#             if(self._id[i]==k):
#                 self._id[i]=self._next_id;  
                
        
#     def parent(self):
#         return [self.parent[i] for i in range(self.n)]

#     def tree(self):
#         return [self._tree[i] for i in range(len(self._tree))]
  