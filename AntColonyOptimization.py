import numpy as np

# 1. Define the Problem: 6 Cities with 2D coordinates (optimal perimeter = 20.0)
CITIES = np.array([
    [0.0, 0.0],  # City 0
    [0.0, 3.0],  # City 1
    [3.0, 5.0],  # City 2
    [6.0, 3.0],  # City 3
    [6.0, 0.0],  # City 4
    [3.0, 0.0]   # City 5
])

def compute_distance_matrix(cities):
    n = len(cities)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = np.linalg.norm(cities[i] - cities[j])
    return dist

def aco_tsp(cities=CITIES, n_ants=15, alpha=1.0, beta=2.5, rho=0.4, tau0=1.0, max_iter=30):
    np.random.seed(42)
    n_cities = len(cities)
    dist_matrix = compute_distance_matrix(cities)
    heuristic = 1.0 / (dist_matrix + 1e-10)
    
    # 2. Initialize Parameters & Pheromone Trails
    pheromone = np.full((n_cities, n_cities), tau0)
    best_tour = None
    best_length = np.inf

    # 5. Iterate
    for it in range(1, max_iter + 1):
        all_tours = []
        all_lengths = []

        # 3. Construct Solutions
        for ant in range(n_ants):
            visited = [0]  # Start at City 0
            unvisited = list(range(1, n_cities))

            while unvisited:
                curr = visited[-1]
                probs = np.array([
                    (pheromone[curr, nxt] ** alpha) * (heuristic[curr, nxt] ** beta)
                    for nxt in unvisited
                ])
                probs /= np.sum(probs)
                next_city = np.random.choice(unvisited, p=probs)
                visited.append(next_city)
                unvisited.remove(next_city)

            visited.append(visited[0])  # Return to origin city
            length = sum(dist_matrix[visited[i], visited[i+1]] for i in range(n_cities))
            
            all_tours.append(visited)
            all_lengths.append(length)

            if length < best_length:
                best_length = length
                best_tour = visited.copy()

        # 4. Update Pheromones
        pheromone *= (1.0 - rho)
        for tour, length in zip(all_tours, all_lengths):
            deposit = 1.0 / length
            for i in range(n_cities):
                u, v = tour[i], tour[i+1]
                pheromone[u, v] += deposit
                pheromone[v, u] += deposit

        if it % 10 == 0 or it == 1:
            print(f"Iteration {it:02d} | Shortest Route Distance = {best_length:.4f} | Route = {best_tour}")

    # 6. Output the Best Solution
    return best_tour, best_length

if __name__ == "__main__":
    best_route, min_dist = aco_tsp()
    print("\n--- Final Output (Ant Colony Optimization - TSP) ---")
    print(f"Optimal Tour Route    : {' -> '.join(map(str, best_route))}")
    print(f"Shortest Tour Distance: {min_dist:.4f}")