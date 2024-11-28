import random
def generate_random_state(board_size):
    return [random.randint(0, board_size-1) for _ in range(board_size)]
def calculate_cost(state):
    cost = 0
    for i in range(len(state)):
        for j in range(i + 1, len(state)):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                cost += 1
    return cost
def hill_climbing_queens(board_size, max_iterations=1000):
    current_state = generate_random_state(board_size)
    current_cost = calculate_cost(current_state)

    for _ in range(max_iterations):
        neighbors = []
        for i in range(board_size):
            for j in range(board_size):
                if current_state[i] != j:
                    neighbor_state = list(current_state)
                    neighbor_state[i] = j
                    neighbors.append((neighbor_state, calculate_cost(neighbor_state)))
        neighbors.sort(key=lambda x: x[1])  # Sort neighbors by cost
        best_neighbor, best_neighbor_cost = neighbors[0]

        if best_neighbor_cost >= current_cost:
            return current_state, current_cost
        current_state, current_cost = best_neighbor, best_neighbor_cost
    return current_state, current_cost
def print_board(state):
    for i in range(len(state)):
        row = ['Q' if j == state[i] else '.' for j in range(len(state))]
        print(' '.join(row))

if __name__ == "__main__":
    board_size = 8
    solution, cost = hill_climbing_queens(board_size)
    print("Final State:")
    print_board(solution)
    print("Cost:", cost)
