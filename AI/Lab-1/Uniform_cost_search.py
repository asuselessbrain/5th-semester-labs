import heapq

def uniform_cost_search(graph, start, goal):
    priority_queue = [(0, start, [])]
    visited = set()

    while priority_queue:
        cost, current_node, path = heapq.heappop(priority_queue)

        if current_node == goal:
            return True, path + [current_node], cost

        visited.add(current_node)

        for neighbor, neighbor_cost in graph.get(current_node, []):
            if neighbor not in visited:
                heapq.heappush(priority_queue, (cost + neighbor_cost, neighbor, path + [current_node]))

    return False, [], 0

# Example graph and usage
graph = {
    'A': [('B', 2), ('C', 5)],
    'B': [('D', 1), ('E', 2)],
    'C': [('F', 4)],
    'D': [('B', 5), ('F', 6), ('E', 2)],
    'E': [('C', 4)],
    'F': [('C', 4)]
}

start_node = 'A'
goal_node = 'F'

found, path, cost = uniform_cost_search(graph, start_node, goal_node)

if found:
    print(f"Goal '{goal_node}' found from '{start_node}' using Uniform Cost Search.")
    print("Path:", path)
    print("Cost:", cost)
else:
    print(f"Goal '{goal_node}' not found from '{start_node}' using Uniform Cost Search.")
