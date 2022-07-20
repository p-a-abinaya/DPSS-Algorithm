import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import copy
from math import atan2, degrees
import sys

from linkedlist import node


grid = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

grid_copy = copy.deepcopy(grid)
 

# start point and goal
start = (2,2)
goal = (17,13)

indx = 0

# TO DISPLAY GRID

fig, ax = plt.subplots(figsize=(12,12))

ax.imshow(grid, cmap=plt.cm.Dark2)

ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)

ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)


#LINKED LIST IMPLEMENTATION: 

class Node:
    def __init__(self, data = (None, None)):
        self.data = data
        self.parent_index = None
        self.index = None
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = Node()
        self.tail = Node()

    def listAppend(self, new_data, parent=None):
        global indx
        
        new_node = Node(new_data)
        if self.head.data == (None, None):
            self.head = new_node
            self.tail = new_node
            new_node.parent_index = parent
        
        else:
            self.tail.next = new_node
            self.tail = new_node
            new_node.parent_index = parent
        new_node.index = indx
        indx+=1

    def printlist(self):
        temp = self.head
        while(temp):
            print(temp.data, end = " ")
            temp = temp.next
        print("")
        
llist = LinkedList()

def isWithinBoundary(a,b):
    if a>=0 and a<20 and b>=0 and b<20:
        return True
    else:
        return False



def possibleNeighbors(current): #Returns possible neighbors 
    neighbors = [(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]
    n1 = copy.deepcopy(neighbors)

    if grid[current[0]-1, current[1]] == 1 and grid[current[0], current[1]-1] == 1:
        del n1[7] 
    if grid[current[0]+1, current[1]] == 1 and grid[current[0], current[1]-1] == 1:
        del n1[5]
    if grid[current[0]+1, current[1]] == 1 and grid[current[0], current[1]+1] == 1:
        del n1[3]
    if grid[current[0]-1, current[1]] == 1 and grid[current[0], current[1]+1] == 1:
        del n1[1]

    n2 = copy.deepcopy(n1)
    k=0
    for i,j in n1:

        x = current[0] + i
        y = current[1] + j
        
        if  isWithinBoundary(x,y) and grid_copy[x,y] == 1:
            del n2[k]         
        else:
            k+=1

    n3 = []
    for i, j in n2:
        n3.append((current[0]+i, current[1]+j))

    n = copy.deepcopy(n3)

    k=0   
    for i,j in n2:
        if not isWithinBoundary(i,j):
            del n[k]
        else:
            k+=1
    return n


def angleBetween(vector1, vector2):
    unit_vector_1 = vector1/np.linalg.norm(vector1)
    unit_vector_2 = vector2/np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = degrees(np.arccos(dot_product))
    return angle


def neighborSort(neighb, current, goal):

    goalv = (goal[0]-current[0], goal[1]-current[1])
    neighbv = []
    for i,j in neighb:
        neighbv.append((i-current[0], j-current[0]))

    anglelist = []
    for k in range(len(neighb)):
        anglelist.append(abs(angleBetween(goalv, neighbv[k])))
    
    anglelist = np.array(anglelist)
    neighb = np.array(neighb)
    inds = anglelist.argsort()
    sortedneighb = neighb[inds]
    sorted = []

    for i,j in sortedneighb:
        sorted.append((i,j))

    return sorted


#DPSS algorithm:
def DPSS(strt, gol):
    global llist
    curr = Node()
    llist.listAppend(strt,-1)
    curr = llist.head

    while curr.data != gol:
        nbhd = possibleNeighbors(curr.data)
        sortednbhd = neighborSort(nbhd, curr.data, gol)

        k=0
        for i,j in sortednbhd:
            if isWithinBoundary(i,j):
                grid_copy[i,j] = 1
                llist.listAppend(sortednbhd[k], curr.index)
                k+=1
        curr = curr.next
   
    path = []
    temp = Node()
    
    if curr.data == gol:
        while curr!= llist.head: 
            path.append(curr.data)
            temp = llist.head
            while temp.index != curr.parent_index:
                temp = temp.next
            curr = temp
        
        path.append(curr.data)
        
    return path

if __name__ == "__main__":
    start = (2,2)
    goal = (17,13)
    route = DPSS(start, goal)
    route = route[::-1]
    x_coords = []
    y_coords = []

    for i in (range(0,len(route))):

        x = route[i][0]
        y = route[i][1]
        x_coords.append(x)
        y_coords.append(y)

    # plot map and path
    fig, ax = plt.subplots(figsize=(20,20))
    ax.imshow(grid, cmap=plt.cm.Dark2)
    ax.scatter(start[1],start[0], marker = "*", color = "yellow", s = 200)
    ax.scatter(goal[1],goal[0], marker = "*", color = "red", s = 200)
    ax.plot(y_coords,x_coords, color = "black")
    plt.show()