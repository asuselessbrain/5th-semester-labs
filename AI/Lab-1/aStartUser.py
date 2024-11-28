import heapq


class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)


def heuristic(node, goal):
    return abs(node.x - goal.x) + abs(node.y - goal.y)


def astar_search(grid, start, goal):
    open_list = []
    closed_set = set()
    heapq.heappush(open_list, start)

    while open_list:
        current_node = heapq.heappop(open_list)

        if current_node.x == goal.x and current_node.y == goal.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next_x, next_y = current_node.x + dx, current_node.y + dy
            if (0 <= next_x < len(grid) and 0 <= next_y < len(grid[0]) and
                    grid[next_x][next_y] != 1 and (next_x, next_y) not in closed_set):
                next_node = Node(next_x, next_y, current_node)
                next_node.g = current_node.g + 1
                next_node.h = heuristic(next_node, goal)
                heapq.heappush(open_list, next_node)
                closed_set.add((next_node.x, next_node.y))

    return None


def create_grid():
    width = int(input("Enter the width of the grid: "))
    height = int(input("Enter the height of the grid: "))
    grid = []
    for _ in range(height):
        row = []
        for _ in range(width):
            row.append(0)
        grid.append(row)
    return grid


def set_obstacles(grid):
    while True:
        obstacle = input("Enter obstacle position as 'x,y' (or press enter to finish): ")
        if obstacle == "":
            break
        x, y = map(int, obstacle.split(','))
        if 0 <= x < len(grid) and 0 <= y < len(grid[0]):
            grid[x][y] = 1
        else:
            print("Invalid position. Position out of grid bounds.")


def main():
    grid = create_grid()
    set_obstacles(grid)
    start_x, start_y = map(int, input("Enter start position as 'x,y': ").split(','))
    goal_x, goal_y = map(int, input("Enter goal position as 'x,y': ").split(','))

    start_node = Node(start_x, start_y)
    goal_node = Node(goal_x, goal_y)

    path = astar_search(grid, start_node, goal_node)

    if path:
        print("Path found:")
        for position in path:
            print(position)
    else:
        print("No path found.")


if __name__ == "__main__":
    main()
