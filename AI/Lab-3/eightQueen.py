class EightQueensCSP:
    def __init__(self, n=8):
        self.n = n

    def is_valid(self, assignment):
        for i in range(len(assignment)):
            for j in range(i + 1, len(assignment)):
                if abs(i - j) == abs(assignment[i] - assignment[j]):
                    return False
        return True

    def backtrack_search(self, assignment=[]):
        if len(assignment) == self.n:
            return assignment
        for value in range(self.n):
            if value not in assignment:
                if self.is_valid(assignment + [value]):
                    result = self.backtrack_search(assignment + [value])
                    if result:
                        return result
        return None

# Example usage for 8 Queens Puzzle:
if __name__ == "__main__":
    n = 8
    puzzle = EightQueensCSP(n)
    solution = puzzle.backtrack_search()
    if solution:
        print("Solution for 8 Queens Puzzle:")
        print(solution)
    else:
        print("No solution found.")
