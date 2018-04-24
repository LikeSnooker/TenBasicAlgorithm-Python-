from collections import deque
#冒泡排序
#思路很简单  每一趟循环将 最大的数 冒泡到最右端
def bubbleSort(Q):
	for s in range(len(Q)-1):
		for m in range(len(Q)-1):
			if Q[m] > Q[m+1]:
				Q[m],Q[m+1] = Q[m+1],Q[m]
	return Q
print("bubbleSort:")
print(bubbleSort([8,2,7,3,9,1,4,5,6]))

#
# 快速排序
# 核心 思路 选一个基准 小的放左边 大的放右边，并对左右分别递归
#
def quickSort(Q):
    if len(Q) <= 1:
        return Q
    left  = [x for x in Q if x < Q[-1]]
    right = [x for x in Q if x > Q[-1]]
    return quickSort(left) + [Q[-1]] + quickSort(right)
S = [8,2,7,3,9,1,4,5,6]
quickSort(S)
print("quicksort:")
print(S)

#
# 归并排序
# 假设有两个已经排序好的数组 [1,3,5,7] [2,4,6] 我们很容易将这两个数组排序 步骤为
# 选取两个数组中最小的元素，将两者之间的更小者放入新数组,直到某个数组为空，
# 然后将另一个数组中的剩余元素全部放入新数组
# [1,3,5,7] 
# [3,4,6]
# []
#   ↓
# [3,5,7]
# [2,4,6]
# [1]
#   ↓
# [3,5,7]
# [4,6]
# [1,3]
#  :
#  
def mergeSort(Q):
    if len(Q) <= 1:
        return Q
    middle = (0 + len(Q) ) >> 1
    left   = mergeSort(Q[0:middle])
    right  = mergeSort(Q[middle:])
    newQ   = [] 
    while left and right:
        if left[0] < right[0]:
            newQ.append(left.pop(0))
        else:
            newQ.append(right.pop(0))
    newQ.extend(left)
    newQ.extend(right)
    return newQ
S1 = [8,2,7,3,9,1,4,5,6]
mergeSort(S1)
print("mergesort")
print(mergeSort(S1))

#  
#  堆排序 利用了 堆结构
#
def leftI(index):
	return (index << 1) + 1
def rightI(index):
	return (index + 1) << 1
def maxheapify(Q,index,size):
	if leftI(index) <= size:
		if Q[leftI(index)] > Q[index]:
			Q[leftI(index)],Q[index] = Q[index],Q[leftI(index)]
		maxheapify(Q,leftI(index),size)
	if rightI(index) <= size:
		if Q[rightI(index)] > Q[index]:
			Q[rightI(index)],Q[index] = Q[index],Q[rightI(index)]
		maxheapify(Q,rightI(index),size)
def buildmaxheap(Q,size):
	for m in range(size):
		maxheapify(Q,0,size)
def heapsort(Q):
	for m in range(len(Q)-1,0,-1):
		buildmaxheap(Q,m)
		Q[0],Q[m] = Q[m],Q[0]
S2 = [8,2,7,3,9,1,4,5,6]
# buildmaxheap(S2,8)
heapsort(S2)
print("heapsort")
print(S2)


###################################################################

a,b,c,d,e,f,g,h = range(8)
N = [
	{b,d},
	{c},
	{f},
	{e},
	{f},
	{g,h},
	{},
	{}
]
#深度优先搜索
def dfs(graph,node):   
	searched,query_queue = set(),[]
	query_queue.append(node)
	while query_queue:
		q_node = query_queue.pop()
		if q_node in searched:
			continue
		searched.add(q_node)
		for neighbor in graph[q_node]:
			query_queue.append(neighbor)
		yield q_node
#广度优先搜索
def bfs(graph,node):
	parents,query_queue = {node:None},deque([node])
	while query_queue:
		q_node = query_queue.popleft()
		for neighbor in graph[q_node]:
			if neighbor in parents:
				continue
			parents[neighbor] = q_node
			query_queue.append(neighbor)
	return parents

print("dfs search")
for dfs_node in dfs(N,a):
	print(dfs_node)
print("bfs search")
for bfs_node in bfs(N,a):
	print(bfs_node)

def mybfs(graph,node):
	explore_queue ,history = deque([node]),set()
	history.add(node)
	while explore_queue:
		wait_explore_node = explore_queue.popleft()
		for neighbor in graph[wait_explore_node]:
			if neighbor in history:
				continue
			history.add(neighbor)
			explore_queue.append(neighbor)
	return history

for my_node in mybfs(N,a):
	print (my_node)

print ("mydfs")
def mydfs(graph,node):
	explore_queue,history = [],set()
	history.add(node)
	explore_queue.append(node)
	while explore_queue:
		cur_node = explore_queue.pop()
		for neighbor in graph[cur_node]:
			if neighbor in history:
				continue
			history.add(neighbor)
			explore_queue.append(neighbor)
			print(cur_node)

mydfs(N,a)
#递归版的深度优先搜索
def rec_dfs(graph,node,history):
	for neighbor in graph[node]:
		if neighbor in history:
			continue
		print(neighbor)
		history.add(neighbor)
		rec_dfs(graph,neighbor,history)
print("rec_dfs")
rec_dfs(N,a,set())

#
# 迪杰斯特拉(dijkstra)算法 
#
import copy
INI   = 999
graph = [[0 ,10,4,8,INI],
         [INI,0,INI,INI,5],
         [INI ,INI,0,2,11],
         [INI,INI,INI,0,3],
         [INI,INI,INI,INI,0]]
def Dijkstra(graph,s,e):
    openList  = [s]
    closeList = [s]
    dists     = copy.copy(graph)
    while openList:
        sorted(openList,key = lambda k:dists[s][k])
        v = openList.pop(0)
        for i in range(len(graph[v])):
            if graph[v][i] == INI:
                continue
            if i in closeList:
                continue
            if dists[s][v] + graph[v][i] < dists[s][i] :
                dists[s][i] = dists[s][v] + graph[v][i]
            openList.append(i)
        closeList.append(v)
    print(dists)

Dijkstra(graph,0,4)


