from collections import deque
import time

# Function to get the graph from the user
def get_graph_from_user():
    graph = {}
    num_nodes = int(input("Enter the number of nodes in the graph: "))
    
    print("For each node, enter the neighbors separated by spaces. If a node has no neighbors, just press Enter.")
    
    for _ in range(num_nodes):
        node = input("\nEnter the name of the node: ")
        neighbors = input(f"Enter the neighbors of {node} (separate by spaces): ").split()
        graph[node] = neighbors

    return graph

# BFS implementation
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    traversal_order = []
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            traversal_order.append(node)
            queue.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
    
    return traversal_order

# DFS implementation
def dfs(graph, start, visited=None, traversal_order=None):
    if visited is None:
        visited = set()
    if traversal_order is None:
        traversal_order = []
    
    visited.add(start)
    traversal_order.append(start)
    
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited, traversal_order)
    
    return traversal_order

# Get the graph from the user
graph = get_graph_from_user()

# Ask the user for the starting node
start_node = input("\nEnter the starting node for traversal: ")

# Measure the time complexity of BFS
bfs_start_time = time.perf_counter()
bfs_traversal = bfs(graph, start_node)
bfs_end_time = time.perf_counter()
bfs_time = bfs_end_time - bfs_start_time

# Measure the time complexity of DFS
dfs_start_time = time.perf_counter()
dfs_traversal = dfs(graph, start_node)
dfs_end_time = time.perf_counter()
dfs_time = dfs_end_time - dfs_start_time

# Display the results
print("\nGraph:", graph)
print("BFS Traversal Order:", bfs_traversal)
print("DFS Traversal Order:", dfs_traversal)
print(f"BFS Time Complexity (Approximate Runtime): {bfs_time:.8f} seconds")
print(f"DFS Time Complexity (Approximate Runtime): {dfs_time:.8f} seconds")


if(bfs_time >dfs_time):
    print("DFS is better than BFS")
else:
    print("BFS is better than DFS")