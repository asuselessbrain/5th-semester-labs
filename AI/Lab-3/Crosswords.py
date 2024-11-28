"""A 10 x 10 Crossword grid is provided, along with a set of words (or names of places)
which need to be filled into the grid. The cells in the grid are initially, either + signs
or – signs. Cells marked with a ‘+’ have to be left as they are. Cells marked with a ‘-‘
need to be filled up with an appropriate character. You are also given an array of words that
need to be filled in Crossword grid."""
ways = 0
def printMatrix(matrix, n):
    for i in range(n):
        print(matrix[i])

def checkHorizontal(x, y, matrix, currentWord):
    n = len(currentWord)

    for i in range(n):
        if matrix[x][y + i] == '#' or matrix[x][y + i] == currentWord[i]:
            matrix[x] = matrix[x][:y + i] + currentWord[i] + matrix[x][y + i + 1:]
        else:

            matrix[0] = "@"
            return matrix
    return matrix
def checkVertical(x, y, matrix, currentWord):
    n = len(currentWord)

    for i in range(n):
        if matrix[x + i][y] == '#' or matrix[x + i][y] == currentWord[i]:
            matrix[x + i] = matrix[x + i][:y] + currentWord[i] + matrix[x + i][y + 1:]
        else:

            matrix[0] = "@"
            return matrix
    return matrix

def solvePuzzle(words, matrix, index, n):
    global ways
    if index < len(words):
        currentWord = words[index]
        maxLen = n - len(currentWord)

        # Loop to check the words that can align vertically.
        for i in range(n):
            for j in range(maxLen + 1):
                temp = checkVertical(j, i, matrix.copy(), currentWord)
                if temp[0] != "@":
                    solvePuzzle(words, temp, index + 1, n)

        # Loop to check the words that can align horizontally.
        for i in range(n):
            for j in range(maxLen + 1):
                temp = checkHorizontal(i, j, matrix.copy(), currentWord)
                if temp[0] != "@":
                    solvePuzzle(words, temp, index + 1, n)
    else:

        print(str(ways + 1) + " way to solve the puzzle ")
        printMatrix(matrix, n)
        print()

        # Increase the ways
        ways += 1
        return


# Driver Code
if __name__ == '__main__':
    # Length of grid
    n1 = 10
    matrix = []
    matrix.append("*#********")
    matrix.append("*#********")
    matrix.append("*#****#***")
    matrix.append("*##***##**")
    matrix.append("*#****#***")
    matrix.append("*#****#***")
    matrix.append("*#****#***")
    matrix.append("*#*######*")
    matrix.append("*#********")
    matrix.append("***#######")

    words = []

    words.append("PUNJAB")
    words.append("JHARKHAND")
    words.append("MIZORAM")
    words.append("MUMBAI")

    ways = 0
    solvePuzzle(words, matrix, 0, n1)
    print("Number of ways to fill the grid is " + str(ways))