import math
import numpy as np

# 1. Define the Problem: Sphere Function f(x) = sum(x_i^2)
def objective_function(x):
    return np.sum(x**2)

def get_levy_step(dim, beta=1.5):
    # Mantegna's algorithm for Levy flights
    num = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    den = math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma_u = (num / den) ** (1 / beta)
    u = np.random.normal(0, sigma_u, size=dim)
    v = np.random.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1 / beta))

def cuckoo_search(n_nests=25, p_a=0.25, dim=2, bounds=(-10.0, 10.0), max_iter=50):
    np.random.seed(42)
    lb, ub = bounds

    # 3. Initialize Population of Nests
    nests = np.random.uniform(lb, ub, (n_nests, dim))
    
    # 4. Evaluate Fitness
    fitness = np.apply_along_axis(objective_function, 1, nests)
    best_idx = np.argmin(fitness)
    best_nest = nests[best_idx].copy()
    best_fit = fitness[best_idx]

    # 7. Iterate
    for it in range(1, max_iter + 1):
        # 5. Generate New Solutions via Levy Flights
        for i in range(n_nests):
            step = get_levy_step(dim)
            new_nest = nests[i] + 0.1 * step * (nests[i] - best_nest)
            new_nest = np.clip(new_nest, lb, ub)
            new_fit = objective_function(new_nest)
            if new_fit < fitness[i]:
                nests[i] = new_nest
                fitness[i] = new_fit

        # 6. Abandon Worst Nests with probability p_a
        abandon_mask = np.random.rand(n_nests, dim) < p_a
        perm1 = nests[np.random.permutation(n_nests)]
        perm2 = nests[np.random.permutation(n_nests)]
        step_size = np.random.rand(n_nests, dim) * (perm1 - perm2)
        new_nests = np.clip(nests + step_size * abandon_mask, lb, ub)
        new_fitness = np.apply_along_axis(objective_function, 1, new_nests)

        improved = new_fitness < fitness
        nests[improved] = new_nests[improved]
        fitness[improved] = new_fitness[improved]

        # Update Global Best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fit:
            best_fit = fitness[min_idx]
            best_nest = nests[min_idx].copy()

        if it % 10 == 0 or it == 1:
            print(f"Iteration {it:02d} | Best Fitness = {best_fit:.8f} | Best Nest = {np.round(best_nest, 5)}")

    # 8. Output the Best Solution
    return best_nest, best_fit

if __name__ == "__main__":
    best_nest, best_fit = cuckoo_search()
    print("\n--- Final Output (Cuckoo Search) ---")
    print(f"Optimal Nest Position : {np.round(best_nest, 6)}")
    print(f"Optimal Fitness Value : {best_fit:.8f}")