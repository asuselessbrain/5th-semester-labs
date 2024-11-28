import networkx as nx
import matplotlib.pyplot as plt

class MapColoringCSP:
    def __init__(self, variables, domains, constraints):
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.graph = self.create_graph()

    def create_graph(self):
        graph = nx.Graph()
        graph.add_nodes_from(self.variables)
        for var1 in self.variables:
            for var2 in self.variables:
                if var1 != var2 and (var2, var1) not in graph.edges:  # Add edges only once
                    if all(self.check_constraint(var1, var2, c1, c2) for c1 in self.domains[var1] for c2 in self.domains[var2]):
                        graph.add_edge(var1, var2)
        return graph

    def check_constraint(self, var1, var2, c1, c2):
        assignment = {var1: c1, var2: c2}
        for constraint in self.constraints:
            if not constraint(assignment):
                return False
        return True

    def visualize_map(self, assignment):
        colors = ['Red', 'Green', 'Blue']
        node_colors = [assignment[node] for node in self.variables]

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color=node_colors, node_size=1000, font_size=12)
        plt.show()

    def is_consistent(self, variable, value, assignment):
        assignment[variable] = value
        for constraint in self.constraints:
            if not constraint(assignment):
                return False
        return True

    def backtrack_search(self, assignment={}):
        if len(assignment) == len(self.variables):
            return assignment
        unassigned_variables = [var for var in self.variables if var not in assignment]
        var = unassigned_variables[0]
        for value in self.domains[var]:
            if self.is_consistent(var, value, assignment):
                new_assignment = assignment.copy()
                new_assignment[var] = value
                result = self.backtrack_search(new_assignment)
                if result:
                    return result
        return None

# Example usage for Map Coloring Problem:
if __name__ == "__main__":
    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    domains = {'WA': ['Red', 'Green', 'Blue'], 'NT': ['Red', 'Green', 'Blue'], 'SA': ['Red', 'Green', 'Blue'],
               'Q': ['Red', 'Green', 'Blue'], 'NSW': ['Red', 'Green', 'Blue'], 'V': ['Red', 'Green', 'Blue'],
               'T': ['Red', 'Green', 'Blue']}
    constraints = [
        lambda assignment: 'WA' not in assignment or 'NT' not in assignment or assignment.get('WA') != assignment.get('NT'),
        lambda assignment: 'WA' not in assignment or 'SA' not in assignment or assignment.get('WA') != assignment.get('SA'),
        lambda assignment: 'NT' not in assignment or 'SA' not in assignment or assignment.get('NT') != assignment.get('SA'),
        lambda assignment: 'NT' not in assignment or 'Q' not in assignment or assignment.get('NT') != assignment.get('Q'),
        lambda assignment: 'SA' not in assignment or 'Q' not in assignment or assignment.get('SA') != assignment.get('Q'),
        lambda assignment: 'SA' not in assignment or 'NSW' not in assignment or assignment.get('SA') != assignment.get('NSW'),
        lambda assignment: 'SA' not in assignment or 'V' not in assignment or assignment.get('SA') != assignment.get('V'),
        lambda assignment: 'SA' not in assignment or 'T' not in assignment or assignment.get('SA') != assignment.get('T'),
        lambda assignment: 'NSW' not in assignment or 'T' not in assignment or assignment.get('NSW') != assignment.get('T'),
        lambda assignment: 'NSW' not in assignment or 'V' not in assignment or assignment.get('NSW') != assignment.get('V')
    ]
    map_coloring = MapColoringCSP(variables, domains, constraints)
    solution = map_coloring.backtrack_search()
    if solution:
        print("Solution for Map Coloring Problem:")
        print(solution)
        map_coloring.visualize_map(solution)
    else:
        print("No solution found.")
