import numpy as np

# 1. Define the Problem: Maximize f(x) = -x^2 + 6x + 20 in range [-5, 10]
# Analytical Maximum is at x = 3.0, where f(3.0) = -9 + 18 + 20 = 29.0
def objective_function(x):
    return -(x**2) + 6 * x + 20

def decode(chromosome, bounds):
    lb, ub = bounds
    max_int = (2**len(chromosome)) - 1
    decimal = int("".join(str(b) for b in chromosome), 2)
    return lb + (decimal / max_int) * (ub - lb)

def tournament_selection(pop, fitness, k=3):
    idx = np.random.choice(len(pop), k, replace=False)
    return pop[idx[np.argmax(fitness[idx])]].copy()

def genetic_algorithm(pop_size=30, chrom_len=16, p_c=0.8, p_m=0.02, generations=40, bounds=(-5.0, 10.0)):
    np.random.seed(42)
    
    # 3. Create Initial Population
    population = np.random.randint(0, 2, size=(pop_size, chrom_len))
    best_x = None
    best_fit = -np.inf

    # 8. Iteration over generations
    for gen in range(1, generations + 1):
        # 4. Evaluate Fitness
        decoded_x = np.array([decode(ind, bounds) for ind in population])
        fitness = np.array([objective_function(x) for x in decoded_x])

        max_idx = np.argmax(fitness)
        if fitness[max_idx] > best_fit:
            best_fit = fitness[max_idx]
            best_x = decoded_x[max_idx]
            elite = population[max_idx].copy()

        if gen % 10 == 0 or gen == 1:
            print(f"Generation {gen:02d} | Best x = {best_x:.5f} | Max Fitness = {best_fit:.6f}")

        new_pop = [elite.copy()]
        while len(new_pop) < pop_size:
            # 5. Selection
            p1 = tournament_selection(population, fitness)
            p2 = tournament_selection(population, fitness)

            # 6. Crossover
            if np.random.rand() < p_c:
                pt = np.random.randint(1, chrom_len - 1)
                c1 = np.concatenate([p1[:pt], p2[pt:]])
                c2 = np.concatenate([p2[:pt], p1[pt:]])
            else:
                c1, c2 = p1.copy(), p2.copy()

            # 7. Mutation
            for child in (c1, c2):
                mask = np.random.rand(chrom_len) < p_m
                child[mask] = 1 - child[mask]
                if len(new_pop) < pop_size:
                    new_pop.append(child)

        population = np.array(new_pop)

    # 9. Output the Best Solution
    return best_x, best_fit

if __name__ == "__main__":
    best_x, best_fit = genetic_algorithm()
    print("\n--- Final Output (Genetic Algorithm) ---")
    print(f"Optimal Solution (x)  : {best_x:.6f}")
    print(f"Maximum Fitness Value : {best_fit:.6f}")