import numpy as np

# 1. Define the Problem: Sphere Function f(x) = sum(x_i^2)
def objective_function(x):
    return np.sum(x**2)

def grey_wolf_optimizer(n_wolves=20, dim=2, bounds=(-10.0, 10.0), max_iter=50):
    np.random.seed(42)
    lb, ub = bounds

    # 3. Initialize Population
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))

    alpha_pos = np.zeros(dim)
    alpha_score = np.inf
    beta_pos = np.zeros(dim)
    beta_score = np.inf
    delta_pos = np.zeros(dim)
    delta_score = np.inf

    # 6. Iterate
    for it in range(1, max_iter + 1):
        # 4. Evaluate Fitness & Update Alpha, Beta, Delta hierarchy
        for i in range(n_wolves):
            wolves[i] = np.clip(wolves[i], lb, ub)
            fitness = objective_function(wolves[i])

            if fitness < alpha_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = alpha_score, alpha_pos.copy()
                alpha_score, alpha_pos = fitness, wolves[i].copy()
            elif fitness < beta_score:
                delta_score, delta_pos = beta_score, beta_pos.copy()
                beta_score, beta_pos = fitness, wolves[i].copy()
            elif fitness < delta_score:
                delta_score, delta_pos = fitness, wolves[i].copy()

        # Linearly decreasing parameter 'a' from 2 to 0
        a = 2.0 - it * (2.0 / max_iter)

        # 5. Update Positions based on Alpha, Beta, and Delta
        for i in range(n_wolves):
            A1 = 2 * a * np.random.rand(dim) - a
            C1 = 2 * np.random.rand(dim)
            X1 = alpha_pos - A1 * np.abs(C1 * alpha_pos - wolves[i])

            A2 = 2 * a * np.random.rand(dim) - a
            C2 = 2 * np.random.rand(dim)
            X2 = beta_pos - A2 * np.abs(C2 * beta_pos - wolves[i])

            A3 = 2 * a * np.random.rand(dim) - a
            C3 = 2 * np.random.rand(dim)
            X3 = delta_pos - A3 * np.abs(C3 * delta_pos - wolves[i])

            wolves[i] = (X1 + X2 + X3) / 3.0

        if it % 10 == 0 or it == 1:
            print(f"Iteration {it:02d} | Alpha Score = {alpha_score:.8f} | Alpha Pos = {np.round(alpha_pos, 5)}")

    # 7. Output the Best Solution
    return alpha_pos, alpha_score

if __name__ == "__main__":
    best_pos, best_score = grey_wolf_optimizer()
    print("\n--- Final Output (Grey Wolf Optimizer) ---")
    print(f"Optimal Alpha Position : {np.round(best_pos, 6)}")
    print(f"Minimum Fitness Score  : {best_score:.8f}")