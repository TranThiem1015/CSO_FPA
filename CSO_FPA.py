import numpy as np
import math
import time
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
try:
    from tabulate import tabulate
except ImportError:
    print("Warning: 'tabulate' library not found. Tables will be printed in basic format.")
    tabulate = None

# ---------------------------
# Utility: read H matrix
# ---------------------------
DEFAULT_H_STRING = """(0.3346837825692431+0.5066204994171795j) (-0.02986783253863495-0.04141545474670731j) (0.17657645829979224+1.8503950663045683j) (-0.07678001611357146-1.066752721185253j)
(-0.5262303695984011+0.12364464304656578j) (-0.9025790875764345-0.48916726325807824j) (0.4340836084103516+0.47680578848504085j) (0.5837517742664149+0.9899351364352043j)
(-0.6479866548987625+0.4478365207166802j) (-0.04269908683070823+0.5002554025589229j) (0.6600008552720403+0.666280522034893j) (-0.7771383338905455+0.0974738384364081j)
(-0.34306604513685574-0.0115486084917641j) (0.09074676431469021-0.25824192747688385j) (-0.4503432904174194+0.40659921527626586j) (-0.4728828417757863-0.9339126306750009j)"""

def read_h_matrix(filename="hkl_array.txt", K=4):
    def parse_complex_list_from_text(text):
        tokens = text.replace("\n", " ").split()
        comps = []
        for t in tokens:
            s = t.strip()
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            c = complex(s.replace("i","j"))
            comps.append(c)
        return comps

    if os.path.isfile(filename):
        try:
            with open(filename, "r") as f:
                txt = f.read()
            comps = parse_complex_list_from_text(txt)
            if len(comps) < K*K:
                raise ValueError("Not enough entries")
            H = np.array(comps[:K*K], dtype=np.complex128).reshape((K, K))
            return H
        except Exception:
            print("Warning: failed to read file, fallback used.")

    comps = parse_complex_list_from_text(DEFAULT_H_STRING)
    H = np.array(comps[:K*K], dtype=np.complex128).reshape((K, K))
    return H

# ---------------------------
# Objective function: sum rate
# ---------------------------
def sum_rate(P, H, sigma2=1.0, debug=False):
    P = np.array(P, dtype=float)
    K = H.shape[0]
    if not np.all(P >= 0):
        print(f"Warning: Negative power detected in P: {P}")
        P = np.clip(P, 0, None)
    rate = 0.0
    for k in range(K):
        signal = P[k] * (np.abs(H[k, k])**2)
        if signal < 0:
            print(f"Warning: Negative signal for user {k}: signal={signal:.6f}, P[{k}]={P[k]:.6f}, |H[{k},{k}]|^2={np.abs(H[k,k])**2:.6f}")
            signal = 0.0
        interference = sigma2 + sum(P[j] * (np.abs(H[k, j])**2) for j in range(K) if j != k)
        interference = max(interference, 0.1)  # Interference floor
        sinr = signal / interference
        sinr = min(max(sinr, 0.0), 10.0)  # SINR cap
        rate += math.log2(1.0 + sinr)
        if debug and rate > 10.0:
            print(f"User {k}: P[{k}]={P[k]:.6f}, Signal={signal:.6f}, Interference={interference:.6f}, SINR={sinr:.6f}, Rate={rate:.6f}")
    if np.isnan(rate) or rate > 10.0:
        print(f"Warning: Excessive sum rate detected: {rate:.6f}")
        return -np.inf
    return rate

# ---------------------------
# Lévy flight
# ---------------------------
def levy_flight(lambda_=1.5, size=1, scale=0.1):
    sigma_u = (math.gamma(1 + lambda_) * math.sin(math.pi * lambda_ / 2) /
              (math.gamma((1 + lambda_) / 2) * lambda_ * 2 ** ((lambda_ - 1) / 2))) ** (1 / lambda_)
    u = np.random.normal(0, sigma_u, size=size)
    v = np.random.normal(0, 1, size=size)
    step = u / (np.abs(v) ** (1.0 / lambda_))
    return scale * step

# ---------------------------
# CSO
# ---------------------------
def cso_power_optimization(K, H, P_max, num_cats=30, max_iter=100,
                           MR=0.2, SMP=4, SRD=0.2, c=2.0, w=0.5, sigma2=1.0):
    start_time = time.time()

    cats = np.random.uniform(0.0, P_max, size=(num_cats, K))
    velocities = np.zeros((num_cats, K))

    # Individual tracking
    individual_x_best = np.copy(cats)
    individual_best_fitness = np.array([sum_rate(cat, H, sigma2, debug=(i==0)) for i, cat in enumerate(cats)])
    individual_best_history = [cat.copy()[None] for cat in cats]
    individual_fitness_history = [np.array([fitness]) for fitness in individual_best_fitness]

    # Global best tracking
    global_best_idx = np.argmax(individual_best_fitness)
    global_best = individual_x_best[global_best_idx].copy()
    global_best_fitness = individual_best_fitness[global_best_idx]
    global_fitness_history = [global_best_fitness]

    # Lưu lịch sử quần thể và fitness cho từng vòng lặp
    population_history = [cats.copy()]
    all_fitness_history = [individual_best_fitness.copy()]

    print(f"\n=== CSO Individual Tracking (P_max={P_max}) ===")
    print(f"Number of cats (NC): {num_cats}")
    print(f"Max iterations: {max_iter}")
    print(f"Initial global best fitness: {global_best_fitness:.6f}")

    # Main CSO loop
    for iteration in range(max_iter):
        num_tracing = max(1, int(num_cats * MR))
        tracing_indices = np.random.choice(num_cats, num_tracing, replace=False)

        for i in range(num_cats):
            if i in tracing_indices:
                r = np.random.uniform(0.0, 1.0, size=K)
                velocities[i] = w * velocities[i] + c * r * (global_best - cats[i])
                if np.any(np.abs(velocities[i]) > 0.05 * P_max):
                    # velocities[i] = np.zeros(K)  # Reset if exceeds limit
                    velocities[i] = np.zeros(K)
                velocities[i] = np.clip(velocities[i], -0.05 * P_max, 0.05 * P_max)
                cats[i] += velocities[i]
                cats[i] = np.clip(cats[i], 0.0, P_max)  # Clip immediately
                # print(f"Cat {i} after tracing: {cats[i]}")  # Debug
            else:
                copies = np.tile(cats[i], (SMP, 1))
                changes = np.random.uniform(-SRD * P_max, SRD * P_max, size=(SMP, K))
                copies += changes
                copies = np.clip(copies, 0.0, P_max)  # Clip immediately
                copy_fitness = np.array([sum_rate(copies[s], H, sigma2, debug=(i==0 and s==0)) for s in range(SMP)])
                copy_fitness[np.isnan(copy_fitness)] = -np.inf

                if np.all(copy_fitness == -np.inf):
                    continue

                probs = np.exp(copy_fitness - np.max(copy_fitness)) / np.sum(np.exp(copy_fitness - np.max(copy_fitness)) + 1e-12)
                selected = np.random.choice(SMP, p=probs)
                cats[i] = copies[selected].copy()  # Explicit copy
                cats[i] = np.clip(cats[i], 0.0, P_max)  # Re-clip after selection
                # print(f"Cat {i} after seeking: {cats[i]}")  # Debug

            current_fitness = sum_rate(cats[i], H, sigma2, debug=(i==0))
            if current_fitness > individual_best_fitness[i]:
                individual_best_fitness[i] = current_fitness
                individual_x_best[i] = cats[i].copy()

            individual_best_history[i] = np.vstack([individual_best_history[i], individual_x_best[i]])
            individual_fitness_history[i] = np.append(individual_fitness_history[i], individual_best_fitness[i])

            # Restart if stuck
            if iteration % 10 == 0 and current_fitness < 0.1 * global_best_fitness:
                print(f"Restarting cat {i} at iteration {iteration}")
                cats[i] = np.random.uniform(0.0, P_max, size=K)
                current_fitness = sum_rate(cats[i], H, sigma2)
                individual_best_fitness[i] = current_fitness
                individual_x_best[i] = cats[i].copy()

        current_global_best_idx = np.argmax(individual_best_fitness)
        if individual_best_fitness[current_global_best_idx] > global_best_fitness:
            global_best = individual_x_best[current_global_best_idx].copy()
            global_best_fitness = individual_best_fitness[current_global_best_idx]

        global_fitness_history.append(global_best_fitness)
        population_history.append(cats.copy())
        all_fitness_history.append(individual_best_fitness.copy())

    end_time = time.time()
    execution_time = end_time - start_time

    return (individual_x_best, individual_best_fitness, individual_best_history,
            individual_fitness_history, global_best, global_best_fitness,
            global_fitness_history, execution_time,
            population_history, all_fitness_history)

# ---------------------------
# FPA
# ---------------------------
def fpa_power_optimization(K, H, P_max, num_flowers=30, max_iter=100,
                           p=0.8, lambda_=1.5, sigma2=1.0):
    start_time = time.time()
    flowers = np.random.uniform(0.0, P_max, size=(num_flowers, K))
    individual_x_best = flowers.copy()
    individual_best_fitness = np.array([sum_rate(f, H, sigma2, debug=(i==0)) for i, f in enumerate(flowers)])
    population_history = [flowers.copy()]
    all_fitness_history = [individual_best_fitness.copy()]

    gb_idx = int(np.argmax(individual_best_fitness))
    global_best = individual_x_best[gb_idx].copy()
    global_best_fitness = float(individual_best_fitness[gb_idx])
    global_fitness_history = [global_best_fitness]
    best_solutions = [{"iter": 0, "x": global_best.copy(), "fitness": global_best_fitness}]

    for it in range(1, max_iter + 1):
        for i in range(num_flowers):
            if np.random.rand() < p:
                L = levy_flight(lambda_, K, scale=0.1)
                flowers[i] = flowers[i] + L * (global_best - flowers[i])
            else:
                eps = np.random.uniform(0, 1, size=K)
                j, k = np.random.choice(num_flowers, 2, replace=False)
                flowers[i] = flowers[i] + eps * (flowers[j] - flowers[k])
            flowers[i] = np.clip(flowers[i], 0.0, P_max)
            current_fitness = sum_rate(flowers[i], H, sigma2, debug=(i==0))
            if current_fitness > individual_best_fitness[i]:
                individual_best_fitness[i] = current_fitness
                individual_x_best[i] = flowers[i].copy()
            if it % 20 == 0 and current_fitness < 0.5 * global_best_fitness:
                flowers[i] = np.random.uniform(0.0, P_max, size=K)
                current_fitness = sum_rate(flowers[i], H, sigma2)
                individual_best_fitness[i] = current_fitness
                individual_x_best[i] = flowers[i].copy()
        idx = int(np.argmax(individual_best_fitness))
        if individual_best_fitness[idx] > global_best_fitness:
            global_best_fitness = float(individual_best_fitness[idx])
            global_best = individual_x_best[idx].copy()
        global_fitness_history.append(global_best_fitness)
        best_solutions.append({"iter": it, "x": global_best.copy(), "fitness": global_best_fitness})
        population_history.append(flowers.copy())
        all_fitness_history.append(individual_best_fitness.copy())

    return global_best, global_best_fitness, global_fitness_history, best_solutions, population_history, all_fitness_history, time.time()-start_time

# ---------------------------
# Population Summary Table
# ---------------------------
def print_population_summary_table(cso_individual_best_history, cso_individual_fitness_history,
                                  fpa_population_histories, fpa_all_fitness_histories,
                                  P_max, K=4, max_iter=100):
    selected_iterations = [0, 20, 40, 60, 80, max_iter]
    print(f"\n=== Population Summary for P_max = {P_max} ===")
    table_data = []
    headers = ['Run', 'Iteration', 'Algorithm', 'Best P[0]', 'Best P[1]', 'Best P[2]', 'Best P[3]', 
               'Best Fitness', 'Avg Fitness', 'Worst Fitness']
    
    num_runs = len(cso_individual_best_history)
    for run_idx in range(num_runs):
        # CSO data
        cso_pop = cso_individual_best_history[run_idx]
        cso_fitness = cso_individual_fitness_history[run_idx]
        for iter_idx in selected_iterations:
            if iter_idx < len(cso_pop):
                best_idx = np.argmax(cso_fitness[iter_idx])
                best_P = cso_pop[iter_idx][best_idx]
                best_fit = cso_fitness[iter_idx][best_idx]
                avg_fit = np.mean(cso_fitness[iter_idx])
                worst_fit = np.min(cso_fitness[iter_idx])
                row = [run_idx + 1, iter_idx, 'CSO'] + [f"{val:.4f}" for val in best_P] + \
                      [f"{best_fit:.4f}", f"{avg_fit:.4f}", f"{worst_fit:.4f}"]
                table_data.append(row)
        
        # FPA data
        fpa_pop = fpa_population_histories[run_idx]
        fpa_fitness = fpa_all_fitness_histories[run_idx]
        for iter_idx in selected_iterations:
            if iter_idx < len(fpa_pop):
                best_idx = np.argmax(fpa_fitness[iter_idx])
                best_P = fpa_pop[iter_idx][best_idx]
                best_fit = fpa_fitness[iter_idx][best_idx]
                avg_fit = np.mean(fpa_fitness[iter_idx])
                worst_fit = np.min(fpa_fitness[iter_idx])
                row = [run_idx + 1, iter_idx, 'FPA'] + [f"{val:.4f}" for val in best_P] + \
                      [f"{best_fit:.4f}", f"{avg_fit:.4f}", f"{worst_fit:.4f}"]
                table_data.append(row)
    
    if tabulate:
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    else:
        print("\t".join(headers))
        for row in table_data:
            print("\t".join(str(x) for x in row))

# ---------------------------
# Multiple runs + statistics
# ---------------------------
def evaluate_algorithms(algorithms, H, K, P_max, num_runs=10, max_iter=100):
    results = {}
    for algo_name, algo_func in algorithms.items():
        all_best_solutions = []
        all_best_fitness = []
        all_histories = []
        all_population_histories = []
        all_fitness_histories = []
        total_time = 0.0
        print(f"\nRunning {algo_name} ...")

        for run in range(num_runs):
            if algo_name == "CSO":
                (individual_x_best, individual_best_fitness, individual_best_history,
                 individual_fitness_history, global_best, best_fitness, hist,
                 elapsed, population_history, all_fitness_history) = algo_func(K, H, P_max, max_iter=max_iter)
                global_best_idx = np.argmax(individual_best_fitness)
                all_best_solutions.append([{"iter": i, "x": x, "fitness": f} for i, x, f in 
                                          zip(range(max_iter + 1), individual_best_history[global_best_idx], hist)])
                all_population_histories.append(individual_best_history)
                all_fitness_histories.append(individual_fitness_history)
            else:  # FPA
                (global_best, best_fitness, hist, best_solutions, pop_history, fit_history,
                 elapsed) = algo_func(K, H, P_max, max_iter=max_iter)
                all_best_solutions.append(best_solutions)
                all_population_histories.append(pop_history)
                all_fitness_histories.append(fit_history)
            
            total_time += elapsed
            all_best_fitness.append(best_fitness)
            all_histories.append(hist)

        avg_time = total_time / num_runs
        best_overall = np.max(all_best_fitness)
        avg_best = np.mean(all_best_fitness)
        std_best = np.std(all_best_fitness)

        results[algo_name] = {
            "runs": all_best_solutions,
            "fitness": all_best_fitness,
            "histories": all_histories,
            "population_histories": all_population_histories,
            "fitness_histories": all_fitness_histories,
            "best_overall": best_overall,
            "avg": avg_best,
            "std": std_best,
            "time_avg": avg_time,
        }

        print_best_solutions(all_best_solutions, algo_name, avg_time)

    print_population_summary_table(
        results["CSO"]["population_histories"], results["CSO"]["fitness_histories"],
        results["FPA"]["population_histories"], results["FPA"]["fitness_histories"],
        P_max, K, max_iter
    )

    return results

def plot_mean_convergence(results, max_iter=100):
    plt.figure(figsize=(8, 6))

    for algo_name, res in results.items():
        histories = res["histories"]
        fitness_histories = res["fitness_histories"]
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
        mean_curve = np.mean(histories, axis=0)
        std_curve = np.std(histories, axis=0)
        # Sửa lỗi: avg_fitness shape (num_runs, min_len), lấy trung bình theo từng iteration
        # avg_fitness: list các mảng (num_cats, min_len) hoặc (min_len,)
        # Chuyển về shape (num_runs, min_len) bằng cách lấy trung bình theo cá thể nếu cần
        # Xử lý shape cho avg_fitness
        avg_fitness = []
        for fh in fitness_histories:
            fh = np.array(fh[:min_len])
            if fh.ndim == 2:
                # fh shape (num_cats, min_len) => lấy trung bình theo cá thể
                avg_fitness.append(np.mean(fh, axis=0))
            elif fh.ndim == 3:
                # fh shape (num_cats, something, min_len) => lấy trung bình theo cá thể và chiều thứ 2
                avg_fitness.append(np.mean(fh, axis=(0,1)))
            else:
                avg_fitness.append(fh)
        avg_fitness = np.array(avg_fitness)
        # Nếu shape là (num_runs, min_len), lấy trung bình theo axis=0
        # Nếu shape là (num_runs, num_cats, min_len), lấy trung bình theo axis=(0,1)
        if avg_fitness.ndim == 2 and avg_fitness.shape[1] == min_len:
            mean_avg_fitness = np.mean(avg_fitness, axis=0)
        elif avg_fitness.ndim == 3 and avg_fitness.shape[2] == min_len:
            mean_avg_fitness = np.mean(avg_fitness, axis=(0,1))
        else:
            mean_avg_fitness = np.mean(avg_fitness)
        # Đảm bảo mean_avg_fitness là 1D có cùng chiều dài với iters
        mean_avg_fitness = np.array(mean_avg_fitness).flatten()
        if mean_avg_fitness.size == 1:
            mean_avg_fitness = np.full_like(iters, mean_avg_fitness[0], dtype=float)
        iters = np.arange(len(mean_curve))

        plt.plot(iters, mean_curve, label=f"{algo_name} Best (mean)", linewidth=2)
        plt.plot(iters, mean_avg_fitness, label=f"{algo_name} Avg (mean)", linestyle='--', linewidth=2)
        plt.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Sum Rate (bits/s/Hz)")
    plt.title("Convergence Comparison: CSO vs FPA (Best and Avg ± Std)")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_convergence_metrics(results, threshold=0.95):
    """Compute simple convergence speed metrics.

    For each algorithm and each run, find the first iteration where the best-so-far
    curve reaches threshold * final_best. Report statistics across runs and AUC.
    """
    metrics = {}
    for algo_name, res in results.items():
        runs = res["histories"]  # list of best history per run
        iters_needed = []
        aucs = []
        final_vals = []
        for h in runs:
            arr = np.array(h)
            final = arr[-1]
            final_vals.append(final)
            target = threshold * final
            # find first index where arr >= target
            idxs = np.where(arr >= target)[0]
            if idxs.size > 0:
                iters_needed.append(int(idxs[0]))
            else:
                iters_needed.append(len(arr)-1)
            # AUC approximate via trapezoid
            aucs.append(float(np.trapz(arr)))

        metrics[algo_name] = {
            "iters_needed_mean": float(np.mean(iters_needed)),
            "iters_needed_median": float(np.median(iters_needed)),
            "iters_needed_std": float(np.std(iters_needed)),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "final_mean": float(np.mean(final_vals))
        }
    return metrics


def print_best_solutions(all_runs, algo_name, avg_time):
    print(f"\n=== {algo_name} Results (average time: {avg_time:.4f} s) ===")
    rows = []
    for run_idx, run_best in enumerate(all_runs, start=1):
        for sol in run_best[::20]:  # Sample every 20 iterations
            rows.append([run_idx, sol["iter"], np.round(sol["x"], 4), f"{sol['fitness']:.4f}"])
    df = pd.DataFrame(rows, columns=["Run", "Iteration", "Best Solution (x)", "Objective f(x)"])
    if tabulate:
        print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
    else:
        print(df.to_string(index=False))

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    K = 4
    P_max = 1.0
    np.random.seed(42)
    H = read_h_matrix("hkl_array.txt", K=K)
    print(f"Current H array:\n{H}")

    # Baseline: Uniform power allocation
    uniform_P = np.full(K, P_max)
    uniform_rate = sum_rate(uniform_P, H)
    print(f"Uniform power allocation rate: {uniform_rate:.4f} bits/s/Hz")

    algorithms = {
        "CSO": lambda K, H, P_max, max_iter, **kwargs: cso_power_optimization(K, H, P_max, max_iter=max_iter),
        "FPA": lambda K, H, P_max, max_iter, **kwargs: fpa_power_optimization(K, H, P_max, max_iter=max_iter)
    }

    results = evaluate_algorithms(algorithms, H, K=K, P_max=P_max, num_runs=10, max_iter=100)
    plot_mean_convergence(results, max_iter=100)
    # Compute and print convergence speed metrics
    metrics = compute_convergence_metrics(results, threshold=0.95)
    print("\n=== Convergence speed metrics (reach 95% of final best) ===")
    for algo, m in metrics.items():
        print(f"{algo}: mean iters={m['iters_needed_mean']:.1f}, median={m['iters_needed_median']:.1f}, std={m['iters_needed_std']:.1f}, AUC_mean={m['auc_mean']:.3f}")
    for algo, res in results.items():
        print(f"\n=== {algo} ===")
        print(f"Best overall: {res['best_overall']:.6f}")
        print(f"Average best: {res['avg']:.6f}")
        print(f"Std dev: {res['std']:.6f}")
        print(f"Average time: {res['time_avg']:.4f}s")