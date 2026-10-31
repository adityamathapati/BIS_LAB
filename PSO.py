import numpy as np

# 1. Define the Problem: Sphere Function f(x) = sum(x_i^2)
# Global Minimum is 0.0 at x = [0, 0]
def objective_function(x):
    return np.sum(x**2)

def pso_algorithm(n_particles=25, dim=2, bounds=(-10.0, 10.0), max_iter=50, w=0.6, c1=1.5, c2=1.5):
    np.random.seed(42)
    lb, ub = bounds

    # 3. Initialize Particles
    positions = np.random.uniform(lb, ub, (n_particles, dim))
    velocities = np.random.uniform(-1.0, 1.0, (n_particles, dim))

    # 4. Evaluate Fitness
    fitness = np.apply_along_axis(objective_function, 1, positions)
    pbest_pos = positions.copy()
    pbest_fit = fitness.copy()

    gbest_idx = np.argmin(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = pbest_fit[gbest_idx]

    # 6. Iterate
    for it in range(1, max_iter + 1):
        r1 = np.random.rand(n_particles, dim)
        r2 = np.random.rand(n_particles, dim)

        # 5. Update Velocities and Positions
        velocities = (w * velocities 
                      + c1 * r1 * (pbest_pos - positions) 
                      + c2 * r2 * (gbest_pos - positions))
        positions = np.clip(positions + velocities, lb, ub)

        # Evaluate new fitness
        fitness = np.apply_along_axis(objective_function, 1, positions)

        # Update Personal Best
        improved = fitness < pbest_fit
        pbest_pos[improved] = positions[improved]
        pbest_fit[improved] = fitness[improved]

        # Update Global Best
        min_idx = np.argmin(pbest_fit)
        if pbest_fit[min_idx] < gbest_fit:
            gbest_fit = pbest_fit[min_idx]
            gbest_pos = pbest_pos[min_idx].copy()

        if it % 10 == 0 or it == 1:
            print(f"Iteration {it:02d} | Global Best Fitness = {gbest_fit:.8f} | Position = {np.round(gbest_pos, 5)}")

    # 7. Output the Best Solution
    return gbest_pos, gbest_fit

if __name__ == "__main__":
    best_pos, best_fit = pso_algorithm()
    print("\n--- Final Output (Particle Swarm Optimization) ---")
    print(f"Optimal Position      : {np.round(best_pos, 6)}")
    print(f"Minimum Fitness Value : {best_fit:.8f}")