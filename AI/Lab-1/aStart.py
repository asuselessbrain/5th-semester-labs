from heapq import heappop, heappush

def astar_search(graph, start, goal, heuristic):

    open_set = [(0, start, [])]  # Priority queue with initial cost, node, and path
    closed_set = set()

    while open_set:
        cost, current_node, path = heappop(open_set)

        if current_node == goal:
            return path + [current_node]

        if current_node not in closed_set:
            closed_set.add(current_node)

            for neighbor, edge_cost in graph.get(current_node, {}).items():
                if neighbor not in closed_set:
                    total_cost = len(path) + edge_cost + heuristic(neighbor, goal)
                    heappush(open_set, (total_cost, neighbor, path + [current_node]))

    return None  # If no path is found

# Example graph represented as an adjacency list with edge costs
example_graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'D': 3, 'E': 5},
    'C': {'F': 2},
    'D': {'G': 4},
    'E': {'G': 1},
    'F': {'G': 3},
    'G': {}
}

# Heuristic function (Euclidean distance)
def euclidean_distance(node, goal):
    x1, y1 = node
    x2, y2 = goal
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

# Example start and goal nodes
start_node = 'A'
goal_node = 'G'

# Example node coordinates for the heuristic function
node_coordinates = {
    'A': (0, 0),
    'B': (1, 1),
    'C': (2, 0),
    'D': (3, 1),
    'E': (3, 0),
    'F': (2, -1),
    'G': (4, 0)
}

# Applying A* search
result = astar_search(example_graph, start_node, goal_node, lambda node, goal: euclidean_distance(node_coordinates[node], node_coordinates[goal]))

print("A* Search Path:", result)
