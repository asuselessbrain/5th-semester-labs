from numpy import array
def beam_search(distances, beta):
    paths_so_far = [[list(), 0]]
    for idx, tier in enumerate(distances):
        if idx > 0:
            print(f'Paths kept after tier {idx - 1}:')
            print(*paths_so_far, sep='\n')
        paths_at_tier = list()
        for i in range(len(paths_so_far)):
            path, distance = paths_so_far[i]
            for j in range(len(tier)):
                path_extended = [path + [j], distance + tier[j]]
                paths_at_tier.append(path_extended)
        paths_ordered = sorted(paths_at_tier, key=lambda element: element[1])
        paths_so_far = paths_ordered[:beta]
        print(f'\nPaths pruned after tier {idx}: ')
        print(*paths_ordered[beta:], sep='\n')

    return paths_so_far
dists = [[1, 3, 2, 5, 8],
         [4, 7, 9, 6, 7]]
dists = array(dists)
best_beta_paths = beam_search(dists, 3)
print('\nThe best \'beta\' paths:')
for beta_path in best_beta_paths:
    print(beta_path)