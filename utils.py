
# coding: utf-8

# In[6]:

import sys
sys.path.insert(0, 'Resources/MagicCube/code/')
import matplotlib.pyplot as plt

from cube import *


# In[7]:

colors = ["White","Yellow","Blue","Green","Orange","Red"]


# In[8]:

def numCompleteFaces(c):
    
    nf = 0
    
    nFaces = 6
    
    for i in range(nFaces):
        if np.sum(c.stickers[i] != c.stickers[i,0,0]) == 0:
            nf += 1
            
    return nf


# In[26]:

class Edge:
    
    def __init__(self,point1,point2):
        
        self.point1 = point1
        self.point2 = point2
        
    def isDone(self,c):
        #Is this edge done in the cube c?
        return c.stickers[self.point1[0],int(c.N/2),int(c.N/2)] == c.stickers[self.point1]             and c.stickers[self.point2[0],int(c.N/2),int(c.N/2)] == c.stickers[self.point2]
        


# In[34]:

class Corner:
    
    def __init__(self,point1,point2,point3):
        
        self.point1 = point1
        self.point2 = point2
        self.point3 = point3
        
    def isDone(self,c):
        
        return c.stickers[self.point1[0],int(c.N/2),int(c.N/2)] == c.stickers[self.point1]             and c.stickers[self.point2[0],int(c.N/2),int(c.N/2)] == c.stickers[self.point2]             and c.stickers[self.point3[0],int(c.N/2),int(c.N/2)] == c.stickers[self.point3]
        


# In[35]:

def computeEdges(c):
    
    edges = np.empty([0])
    
    for n in range(1,c.N-1):
        edges = np.append(edges,Edge((0,0,n),(5,n,-1))) #cube-ligne-colonne
        edges = np.append(edges,Edge((0,n,-1),(3,n,-1)))
        edges = np.append(edges,Edge((0,n,0),(2,n,-1)))
        edges = np.append(edges,Edge((0,-1,n),(4,n,-1)))
        edges = np.append(edges,Edge((3,-1,n),(5,0,n)))
        edges = np.append(edges,Edge((3,0,n),(4,-1,n)))
        edges = np.append(edges,Edge((4,0,n),(2,-1,n)))
        edges = np.append(edges,Edge((2,0,n),(5,-1,n)))
        edges = np.append(edges,Edge((1,n,-1),(2,n,0)))
        edges = np.append(edges,Edge((1,n,0),(3,n,0)))
        edges = np.append(edges,Edge((1,0,n),(5,n,0)))
        edges = np.append(edges,Edge((1,-1,n),(4,n,0)))
        
        
    return edges


# In[36]:

def computeCorners(c):
    
    corners = np.empty([0])
    
    corners = np.append(corners,Corner((0,0,0),(2,0,-1),(5,-1,-1)))
    corners = np.append(corners,Corner((2,0,0),(1,0,-1),(5,-1,0)))
    corners = np.append(corners,Corner((1,0,0),(3,-1,0),(5,0,0)))
    corners = np.append(corners,Corner((1,-1,0),(4,-1,0),(3,0,0)))
    corners = np.append(corners,Corner((1,-1,-1),(2,-1,0),(4,0,0)))
    corners = np.append(corners,Corner((0,0,-1),(5,0,-1),(3,-1,-1)))
    corners = np.append(corners,Corner((0,-1,-1),(4,-1,-1),(3,0,-1)))
    corners = np.append(corners,Corner((0,-1,0),(4,0,-1),(2,-1,-1)))
        
        
    return corners


# In[37]:

def numCompleteEdges(c,edges):
    
    ne = 0
    
    for e in edges:
        if e.isDone(c):
            ne += 1
            
    return ne


# In[41]:

def numCompleteCorners(c,corners):
    
    nc = 0
    
    for co in corners:
        if co.isDone(c):
            nc += 1
            
    return nc


# In[42]:

def randomMove(c,number): # Random but does not move the centers. Not neccesary at this point.
    
    for t in range(number):
        f = c.dictface[np.random.randint(6)]
        
        l = int(c.N/2)
        while l == int(c.N/2):
            l = np.random.randint(c.N)
            
        d = 2*np.random.randint(2)-1
        c.move(f, l, d)


