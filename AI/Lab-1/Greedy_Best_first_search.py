import heapq

def greedy_best_first_search(graph, start, goal, heuristic):
    priority_queue = [(heuristic[start], start, [start])]

    while priority_queue:
        _, node, path = heapq.heappop(priority_queue)

        if node == goal:
            return path

        for neighbor in graph[node]:
            if neighbor not in path:
                heapq.heappush(priority_queue, (heuristic[neighbor], neighbor, path + [neighbor]))

# Example usage:
# Define a graph as an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

start_node = 'A'
goal_node = 'G'
heuristic = {
    'A': 2,
    'B': 4,
    'C': 3,
    'D': 5,
    'E': 3,
    'F': 2,
    'G': 0
}

result = greedy_best_first_search(graph, start_node, goal_node, heuristic)
print("Greedy Best-First Search Path:", result)
