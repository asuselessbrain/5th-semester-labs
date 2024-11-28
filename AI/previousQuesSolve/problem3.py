import heapq

# Function to take user input for the graph and heuristic values
def input_graph_and_heuristics():
    graph = {}
    heuristics = {}
    
    # Input number of nodes
    num_nodes = int(input("Enter the number of nodes: "))
    
    # Input nodes
    nodes = input(f"Enter the names of the {num_nodes} nodes (space-separated): ").split()
    
    # Input edges for the graph
    print("Enter the edges in the format 'Node1 Node2 Cost'. Type 'done' when finished:")
    while True:
        edge_input = input()
        if edge_input.lower() == "done":
            break
        node1, node2, cost = edge_input.split()
        cost = int(cost)
        
        if node1 not in graph:
            graph[node1] = []
        if node2 not in graph:
            graph[node2] = []
        
        graph[node1].append((node2, cost))
        graph[node2].append((node1, cost))  # Assuming the graph is undirected
    
    # Input heuristic values
    print("Enter the heuristic value for each node:")
    for node in nodes:
        heuristics[node] = int(input(f"Heuristic value for {node}: "))
    
    return graph, heuristics

# A* search algorithm
def a_star_search(graph, heuristics, start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristics[start], 0, start, [start]))
    
    g_costs = {start: 0}
    
    while open_list:
        f_cost, g_cost, current_node, path = heapq.heappop(open_list)
        
        if current_node == goal:
            return path, g_cost
        
        for neighbor, cost in graph.get(current_node, []):
            tentative_g_cost = g_cost + cost
            f_cost = tentative_g_cost + heuristics[neighbor]
            
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                heapq.heappush(open_list, (f_cost, tentative_g_cost, neighbor, path + [neighbor]))
    
    return None, float('inf')

# Main function to run the A* search
def main():
    graph, heuristics = input_graph_and_heuristics()
    
    start_node = input("Enter the start node: ")
    goal_node = input("Enter the goal node: ")
    
    path, total_cost = a_star_search(graph, heuristics, start_node, goal_node)
    
    if path:
        print("Path found:", " -> ".join(path))
        print("Total cost:", total_cost)
    else:
        print("No path found from", start_node, "to", goal_node)

# Run the main function
main()
