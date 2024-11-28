import random

# Define the board size (8x8 for the 8-Queens problem)
N = 8

# Function to generate a random initial state
def generate_initial_state():
    return [random.randint(0, N - 1) for _ in range(N)]

# Function to calculate the number of conflicts for a given state
def calculate_conflicts(state):
    conflicts = 0
    for i in range(N):
        for j in range(i + 1, N):
            if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1
    return conflicts

# Function to perform the hill climbing search
def hill_climbing(state):
    current_state = state[:]
    current_conflicts = calculate_conflicts(current_state)
    
    while True:
        neighbors = []
        for col in range(N):
            for row in range(N):
                if row != current_state[col]:
                    new_state = current_state[:]
                    new_state[col] = row
                    neighbors.append((new_state, calculate_conflicts(new_state)))
        
        # Find the neighbor with the fewest conflicts
        best_neighbor, best_conflicts = min(neighbors, key=lambda x: x[1])
        
        # If no improvement, return the current state
        if best_conflicts >= current_conflicts:
            break
        
        current_state, current_conflicts = best_neighbor, best_conflicts
    
    return current_state, current_conflicts

# Function to print the board for visualization
def print_board(state):
    for row in range(N):
        line = ""
        for col in range(N):
            if state[col] == row:
                line += "1 "
            else:
                line += "0 "
        print(line)
    print()

# Main function to solve the 8-Queens problem with random restarts
def solve_8_queens_with_restarts(max_restarts=100):
    for restart in range(max_restarts):
        initial_state = generate_initial_state()
        solution_state, conflicts = hill_climbing(initial_state)
        
        if conflicts == 0:
            print("Solution found after", restart + 1, "restarts:")
            print_board(solution_state)
            return solution_state
    
    print("No solution found after", max_restarts, "restarts.")
    return None

# Solve the 8-Queens problem
solve_8_queens_with_restarts()
