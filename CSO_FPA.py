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
# /**
#  * @brief Read or construct the channel matrix H.
#  *
#  * Description:
#  *   Attempts to parse a KxK complex channel matrix from a text file
#  *   (entries like `(a+bj)` separated by whitespace). If the file is
#  *   missing or malformed, falls back to an embedded default string.
#  *
#  * Inputs:
#  *   - filename (str): path to a text file containing at least K*K complex entries.
#  *   - K (int): desired square matrix dimension (number of users).
#  *
#  * Outputs:
#  *   - returns numpy.ndarray(shape=(K,K), dtype=complex128) representing H.
#  *   - on failure to parse file, returns H built from internal DEFAULT_H_STRING.
#  */

    def parse_complex_list_from_text(text):
        # Normalize whitespace and split into tokens
        tokens = text.replace("\n", " ").split()
        comps = []
        for t in tokens:
            s = t.strip()
            # strip surrounding parentheses if present
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            # allow 'i' or 'j' for imaginary unit; complex() requires 'j'
            c = complex(s.replace("i", "j"))
            comps.append(c)
        return comps

    # Try reading from file first. If reading/parsing fails, fall back to the
    # DEFAULT_H_STRING embedded in the script (useful for quick testing).
    if os.path.isfile(filename):
        try:
            with open(filename, "r") as f:
                txt = f.read()
            comps = parse_complex_list_from_text(txt)
            if len(comps) < K * K:
                raise ValueError("Not enough entries in file for H matrix")
            H = np.array(comps[:K * K], dtype=np.complex128).reshape((K, K))
            return H
        except Exception:
            print("Warning: failed to read file, falling back to default H string.")

    # Fallback: parse DEFAULT_H_STRING included in repository
    comps = parse_complex_list_from_text(DEFAULT_H_STRING)
    H = np.array(comps[:K * K], dtype=np.complex128).reshape((K, K))
    return H

# ---------------------------
# Objective function: sum rate
# ---------------------------
def sum_rate(P, H, sigma2=1.0, debug=False):
# /**
#  * @brief Compute the sum-rate objective for a power allocation P.
#  *
#  * Description:
#  *   Computes sum_k log2(1 + SINR_k) where SINR_k = P[k]*|H[k,k]|^2 / (sigma2 + sum_{j!=k} P[j]*|H[k,j]|^2).
#  *
#  * Inputs:
#  *   - P (iterable or np.ndarray, shape (K,)): non-negative transmit powers per user.
#  *   - H (np.ndarray, shape (K,K), complex): channel matrix.
#  *   - sigma2 (float): noise power (default 1.0).
#  *   - debug (bool): if True, emits diagnostic prints for large values.
#  *
#  * Outputs:
#  *   - returns float: computed sum-rate (bits/s/Hz). Returns -inf on NaN or unreasonable values.
#  */
    P = np.array(P, dtype=float)
    K = H.shape[0]
    # clip negative powers and warn
    if not np.all(P >= 0):
        print(f"Warning: Negative power detected in P: {P}")
        P = np.clip(P, 0, None)

    rate = 0.0
    for k in range(K):
        # Signal power for user k
        signal = P[k] * (np.abs(H[k, k]) ** 2)
        if signal < 0:
            # defensive: numerical issues may cause a tiny negative number
            print(f"Warning: Negative signal for user {k}: signal={signal:.6f}")
            signal = 0.0

        # Total interference plus noise (exclude user k)
        interference = sigma2 + sum(P[j] * (np.abs(H[k, j]) ** 2) for j in range(K) if j != k)
        interference = max(interference, 0.1)  # small floor to avoid division blowup

        # SINR and safety caps
        sinr = signal / interference
        sinr = min(max(sinr, 0.0), 10.0)

        rate += math.log2(1.0 + sinr)
        if debug and rate > 10.0:
            print(f"User {k}: P[{k}]={P[k]:.6f}, Signal={signal:.6f}, Interference={interference:.6f}, SINR={sinr:.6f}, Rate={rate:.6f}")

    # guard against numerical explosions: return -inf if rate is NaN or unreasonably large
    if np.isnan(rate) or rate > 10.0:
        print(f"Warning: Excessive sum rate detected: {rate:.6f}")
        return -np.inf
    return rate

# ---------------------------
# Lévy flight
# ---------------------------
def levy_flight(lambda_=1.5, size=1, scale=0.1):
# /**
#  * @brief Generate Lévy-distributed steps (Mantegna's algorithm).
#  *
#  * Inputs:
#  *   - lambda_ (float): Lévy tail exponent (>0).
#  *   - size (int or tuple): number of samples or array shape to generate.
#  *   - scale (float): multiplicative factor applied to sampled steps.
#  *
#  * Outputs:
#  *   - returns numpy.ndarray shaped `size` containing Lévy steps (dtype float).
#  */
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
# /**
#  * @brief Cat Swarm Optimization (CSO) variant for power allocation.
#  *
#  * Description:
#  *   Population-based metaheuristic. Tracks per-individual personal bests and
#  *   a global best while alternating tracing (velocity) and seeking (local) modes.
#  *
#  * Inputs:
#  *   - K (int): problem dimension (number of users).
#  *   - H (np.ndarray): complex channel matrix (K,K).
#  *   - P_max (float): per-user max transmit power.
#  *   - num_cats (int): population size.
#  *   - max_iter (int): number of iterations.
#  *   - MR, SMP, SRD, c, w, sigma2: algorithm/hyperparameters (see code defaults).
#  *
#  * Outputs:
#  *   - returns tuple: (individual_x_best, individual_best_fitness,
#  *       individual_best_history, individual_fitness_history,
#  *       global_best, global_best_fitness, global_fitness_history,
#  *       execution_time, population_history, all_fitness_history)
#  *     where:
#  *       - individual_x_best: np.ndarray(num_cats, K) of personal best vectors
#  *       - individual_best_fitness: np.ndarray(num_cats,) personal best fitness
#  *       - individual_best_history: list[num_cats] of arrays (iter+1, K)
#  *       - individual_fitness_history: list[num_cats] of 1D arrays of personal-best fitness per iter
#  *       - global_best: np.ndarray(K,) final global best vector
#  *       - global_best_fitness: float
#  *       - global_fitness_history: list of best fitness per iter
#  *       - execution_time: float seconds
#  *       - population_history: list of population arrays per iter
#  *       - all_fitness_history: list of per-iteration personal-best fitness arrays
#  */
    start_time = time.time()

    # Initialize cat positions and velocities. Each cat is a candidate power
    # allocation vector of length K; velocities are used when in 'tracing' mode.
    cats = np.random.uniform(0.0, P_max, size=(num_cats, K))
    velocities = np.zeros((num_cats, K))

    # Individual best tracking: for every cat keep its personal best position and
    # best fitness seen so far. We also keep per-cat history arrays to later
    # produce population snapshots for reporting.
    individual_x_best = np.copy(cats)
    individual_best_fitness = np.array([sum_rate(cat, H, sigma2, debug=(i == 0)) for i, cat in enumerate(cats)])
    # individual_best_history: list (length=num_cats) of arrays shaped (1, K) to
    # which we will append new personal-best positions per iteration.
    individual_best_history = [cat.copy()[None] for cat in cats]
    # individual_fitness_history: list (length=num_cats) of 1D arrays storing
    # that cat's personal-best fitness vs iteration.
    individual_fitness_history = [np.array([fitness]) for fitness in individual_best_fitness]

    # Global best (best among all cats)
    global_best_idx = np.argmax(individual_best_fitness)
    global_best = individual_x_best[global_best_idx].copy()
    global_best_fitness = individual_best_fitness[global_best_idx]
    global_fitness_history = [global_best_fitness]

    # Store per-iteration snapshots of the whole population (positions) and
    # the personal-best fitness of each individual. These lists will be useful
    # later for building population summary tables.
    population_history = [cats.copy()]
    all_fitness_history = [individual_best_fitness.copy()]

    # Print a concise header for tracing runs
    print(f"\n=== CSO Individual Tracking (P_max={P_max}) ===")
    print(f"Number of cats (NC): {num_cats}")
    print(f"Max iterations: {max_iter}")
    print(f"Initial global best fitness: {global_best_fitness:.6f}")

    # Main CSO loop: each iteration we randomly choose a subset of cats to
    # perform tracing (velocity-based move towards the global best) while the
    # others perform the seeking mode (local sampling around current position).
    for iteration in range(max_iter):
        num_tracing = max(1, int(num_cats * MR))
        tracing_indices = np.random.choice(num_cats, num_tracing, replace=False)

        for i in range(num_cats):
            if i in tracing_indices:
                # Tracing mode: velocity update towards global_best
                r = np.random.uniform(0.0, 1.0, size=K)
                velocities[i] = w * velocities[i] + c * r * (global_best - cats[i])
                # If velocity unexpectedly large, zero it to avoid runaway steps.
                if np.any(np.abs(velocities[i]) > 0.05 * P_max):
                    velocities[i] = np.zeros(K)
                velocities[i] = np.clip(velocities[i], -0.05 * P_max, 0.05 * P_max)
                cats[i] += velocities[i]
                cats[i] = np.clip(cats[i], 0.0, P_max)  # keep valid power allocations
            else:
                # Seeking mode: create SMP candidate copies with small random
                # perturbations (controlled by SRD) and choose probabilistically.
                copies = np.tile(cats[i], (SMP, 1))
                changes = np.random.uniform(-SRD * P_max, SRD * P_max, size=(SMP, K))
                copies += changes
                copies = np.clip(copies, 0.0, P_max)
                copy_fitness = np.array([sum_rate(copies[s], H, sigma2, debug=(i == 0 and s == 0)) for s in range(SMP)])
                copy_fitness[np.isnan(copy_fitness)] = -np.inf

                if np.all(copy_fitness == -np.inf):
                    # All candidates were invalid; skip update for this cat.
                    continue

                # Softmax selection across candidate copies to bias towards
                # higher-fitness local samples while allowing exploration.
                probs = np.exp(copy_fitness - np.max(copy_fitness)) / np.sum(np.exp(copy_fitness - np.max(copy_fitness)) + 1e-12)
                selected = np.random.choice(SMP, p=probs)
                cats[i] = copies[selected].copy()
                cats[i] = np.clip(cats[i], 0.0, P_max)

            # Evaluate and update personal best if current position improved
            current_fitness = sum_rate(cats[i], H, sigma2, debug=(i == 0))
            if current_fitness > individual_best_fitness[i]:
                individual_best_fitness[i] = current_fitness
                individual_x_best[i] = cats[i].copy()

            # Append histories for this individual
            individual_best_history[i] = np.vstack([individual_best_history[i], individual_x_best[i]])
            individual_fitness_history[i] = np.append(individual_fitness_history[i], individual_best_fitness[i])

            # Restart logic: occasionally resample the cat's current position if
            # it's performing very poorly relative to the global best. When we
            # resample we do not overwrite the stored personal best unless the
            # new candidate is strictly better. This prevents sudden drops in
            # the recorded historical personal-best fitness arrays.
            if iteration % 10 == 0 and current_fitness < 0.1 * global_best_fitness:
                new_candidate = np.random.uniform(0.0, P_max, size=K)
                new_fitness = sum_rate(new_candidate, H, sigma2)
                if new_fitness > individual_best_fitness[i]:
                    individual_best_fitness[i] = new_fitness
                    individual_x_best[i] = new_candidate.copy()
                # Move the cat to the new candidate even if it's worse (exploration)
                cats[i] = new_candidate

        # Update global best after all cats moved this iteration
        current_global_best_idx = np.argmax(individual_best_fitness)
        if individual_best_fitness[current_global_best_idx] > global_best_fitness:
            global_best = individual_x_best[current_global_best_idx].copy()
            global_best_fitness = individual_best_fitness[current_global_best_idx]

        # Record iteration-level histories
        global_fitness_history.append(global_best_fitness)
        population_history.append(cats.copy())
        all_fitness_history.append(individual_best_fitness.copy())

    end_time = time.time()
    execution_time = end_time - start_time

    # Return a tuple containing per-individual bests, histories, global best and
    # timing, plus per-iteration population snapshots for downstream reporting.
    return (individual_x_best, individual_best_fitness, individual_best_history,
            individual_fitness_history, global_best, global_best_fitness,
            global_fitness_history, execution_time,
            population_history, all_fitness_history)

# ---------------------------
# FPA
# ---------------------------
def fpa_power_optimization(K, H, P_max, num_flowers=30, max_iter=100,
                                                     p=0.8, lambda_=1.5, sigma2=1.0, levy_scale=0.1, elitism=2):
# /**
#  * @brief Flower Pollination Algorithm (FPA) variant for power allocation.
#  *
#  * Description:
#  *   Population metaheuristic mixing Lévy-flight global moves and local pollination
#  *   updates. Tracks per-flower personal bests and a global best.
#  *
#  * Inputs:
#  *   - K (int): problem dimension.
#  *   - H (np.ndarray): complex channel matrix (K,K).
#  *   - P_max (float): per-user max power.
#  *   - num_flowers (int): population size.
#  *   - max_iter (int): number of iterations.
#  *   - p (float): probability of global pollination (0..1).
#  *   - lambda_, levy_scale: Lévy flight parameters.
#  *   - sigma2, elitism: objective/noise and reserved options.
#  *
#  * Outputs:
#  *   - returns (global_best, global_best_fitness, global_fitness_history,
#  *       best_solutions, population_history, all_fitness_history, elapsed)
#  *     where best_solutions is a list of dicts {'iter', 'x', 'fitness'}.
#  */
    start_time = time.time()
    # Initialize flower positions (candidate power allocations)
    flowers = np.random.uniform(0.0, P_max, size=(num_flowers, K))
    individual_x_best = flowers.copy()
    individual_best_fitness = np.array([sum_rate(f, H, sigma2, debug=(i == 0)) for i, f in enumerate(flowers)])
    population_history = [flowers.copy()]
    all_fitness_history = [individual_best_fitness.copy()]

    # Global best initialization
    gb_idx = int(np.argmax(individual_best_fitness))
    global_best = individual_x_best[gb_idx].copy()
    global_best_fitness = float(individual_best_fitness[gb_idx])
    global_fitness_history = [global_best_fitness]
    best_solutions = [{"iter": 0, "x": global_best.copy(), "fitness": global_best_fitness}]

    # Main FPA loop: at each iteration perform either global pollination
    # (Lévy-flight-guided moves towards the global best) or local pollination
    # (small perturbations based on two randomly chosen flowers).
    for it in range(1, max_iter + 1):
        for i in range(num_flowers):
            if np.random.rand() < p:
                # global pollination using Lévy flights
                L = levy_flight(lambda_, K, scale=levy_scale)
                flowers[i] = flowers[i] + L * (global_best - flowers[i])
            else:
                # local pollination: random perturbation influenced by two
                # other flowers
                eps = np.random.uniform(0, 1, size=K)
                j, k = np.random.choice(num_flowers, 2, replace=False)
                flowers[i] = flowers[i] + eps * (flowers[j] - flowers[k])

            # keep flower within feasible bounds
            flowers[i] = np.clip(flowers[i], 0.0, P_max)
            current_fitness = sum_rate(flowers[i], H, sigma2, debug=(i == 0))

            # Update personal best when improved
            if current_fitness > individual_best_fitness[i]:
                individual_best_fitness[i] = current_fitness
                individual_x_best[i] = flowers[i].copy()

                # Restart-like mechanism for low-performing individuals every
                # few iterations: resample a new flower but only update stored
                # personal best if it's an improvement (to avoid dropping
                # historical bests).
                if it % 20 == 0 and current_fitness < 0.5 * global_best_fitness:
                    new_flower = np.random.uniform(0.0, P_max, size=K)
                    new_fit = sum_rate(new_flower, H, sigma2)
                    if new_fit > individual_best_fitness[i]:
                        individual_best_fitness[i] = new_fit
                        individual_x_best[i] = new_flower.copy()
                    flowers[i] = new_flower

        # Update global best after evaluating all flowers for this iteration
        idx = int(np.argmax(individual_best_fitness))
        if individual_best_fitness[idx] > global_best_fitness:
            global_best_fitness = float(individual_best_fitness[idx])
            global_best = individual_x_best[idx].copy()

        global_fitness_history.append(global_best_fitness)
        best_solutions.append({"iter": it, "x": global_best.copy(), "fitness": global_best_fitness})
        population_history.append(flowers.copy())
        all_fitness_history.append(individual_best_fitness.copy())

    return global_best, global_best_fitness, global_fitness_history, best_solutions, population_history, all_fitness_history, time.time() - start_time


def plot_diagnostics(results, max_iter=100, sample_runs=3):
# /**
#  * @brief Plot diagnostic figures for a results dictionary.
#  *
#  * Inputs:
#  *   - results (dict): aggregated results mapping algorithm->data as returned by evaluate_algorithms.
#  *   - max_iter (int): plotting horizon (not strictly enforced here).
#  *   - sample_runs (int): number of individual runs to plot per algorithm.
#  *
#  * Outputs:
#  *   - produces matplotlib figures (displayed or saved externally).
#  */
    import matplotlib.pyplot as plt
    for algo_name, res in results.items():
        runs_hist = res["histories"]  # list of best-so-far curves
        pop_hist = res["fitness_histories"]  # list of per-run population fitness arrays
        n_runs = len(runs_hist)
        sample_ix = list(range(min(sample_runs, n_runs)))

        plt.figure(figsize=(10, 4))
        # per-run best-so-far
        plt.subplot(1, 2, 1)
        for i in sample_ix:
            h = np.array(runs_hist[i])
            plt.plot(np.arange(len(h)), h, label=f"run{i+1}")
        plt.title(f"{algo_name} best-so-far (sample {len(sample_ix)})")
        plt.xlabel("Iter")
        plt.ylabel("Best so far")
        plt.legend()

        # population final histogram
        plt.subplot(1, 2, 2)
        finals = []
        for i in range(n_runs):
            fh = np.array(pop_hist[i])
            # fh expected shape: list of per-iteration arrays; take last iteration
            last = fh[-1]
            # last may be 1D array of individuals
            finals.extend(np.ravel(last).tolist())
        plt.hist(finals, bins=12)
        plt.title(f"{algo_name} final individual fitness distribution ({n_runs} runs)")
        plt.xlabel("Fitness")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

# ---------------------------
# Population Summary Table
# ---------------------------
def get_snapshot_iterations(max_iter, num_snapshots=5):
# /**
#  * @brief Compute snapshot iteration indices for concise tables.
#  *
#  * Inputs:
#  *   - max_iter (int): maximum iteration index.
#  *   - num_snapshots (int): desired number of snapshots (including start and end).
#  *
#  * Outputs:
#  *   - returns sorted list of unique integer iteration indices.
#  */
    if num_snapshots <= 2:
        return [0, max_iter]
    # generate num_snapshots points including 0 and max_iter
    return sorted(list({int(round(x)) for x in np.linspace(0, max_iter, num_snapshots)}))


def print_population_summary_table(cso_individual_best_history, cso_individual_fitness_history,
                      fpa_population_histories, fpa_all_fitness_histories,
                      P_max, K=4, max_iter=100, selected_iterations=None, num_snapshots=5,
                      output_path=None):
# /**
#  * @brief Print and optionally save a compact population snapshot table.
#  *
#  * Inputs:
#  *   - cso_individual_best_history: list (num_runs) of per-cat best-position histories
#  *   - cso_individual_fitness_history: list (num_runs) of per-cat personal-best fitness arrays
#  *   - fpa_population_histories: list (num_runs) of per-iteration populations for FPA
#  *   - fpa_all_fitness_histories: list (num_runs) of per-iteration fitness arrays for FPA
#  *   - P_max, K, max_iter, selected_iterations, num_snapshots, output_path: display/control params
#  *
#  * Outputs:
#  *   - prints a compact table to stdout and optionally writes CSV/TXT to `output_path`.
#  */
    if selected_iterations is None:
        selected_iterations = get_snapshot_iterations(max_iter, num_snapshots=num_snapshots)
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
    
    # Print to console
    if tabulate:
        printed = tabulate(table_data, headers=headers, tablefmt='grid')
        print(printed)
    else:
        printed_lines = ["\t".join(headers)] + ["\t".join(str(x) for x in row) for row in table_data]
        printed = "\n".join(printed_lines)
        print(printed)

    # Optionally save to files
    if output_path:
        # Ensure directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # Save CSV (one row per table row)
        try:
            df = pd.DataFrame(table_data, columns=headers)
            csv_path = output_path if output_path.endswith('.csv') else output_path + '.csv'
            df.to_csv(csv_path, index=False)
            # Also save a plain text tab-separated file for easy inclusion in reports
            txt_path = output_path if output_path.endswith('.txt') else output_path + '.txt'
            with open(txt_path, 'w') as f:
                f.write(printed)
            print(f"Population summary written: {csv_path}, {txt_path}")
        except Exception as e:
            print(f"Warning: failed to write population summary to {output_path}: {e}")


# ---------------------------
# Multiple runs + statistics
# ---------------------------
def evaluate_algorithms(algorithms, H, K, P_max, num_runs=10, max_iter=100):
# /**
#  * @brief Run each algorithm multiple times and collect statistics and histories.
#  *
#  * Inputs:
#  *   - algorithms (dict): mapping name->callable(K,H,P_max,max_iter=...)
#  *   - H, K, P_max: problem definition arguments
#  *   - num_runs (int): repeats per algorithm
#  *   - max_iter (int): iteration budget per run
#  *
#  * Outputs:
#  *   - returns dict with per-algorithm aggregated results including:
#  *       runs, fitness, histories, population_histories, fitness_histories,
#  *       best_overall, avg, std, time_avg
#  */
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
                # population_history and all_fitness_history returned by CSO are
                # per-iteration snapshots (list length ~= max_iter+1). We must
                # store those so the population summary table can index by
                # iteration. individual_best_history is per-cat history and
                # not suitable here.
                all_population_histories.append(population_history)
                all_fitness_histories.append(all_fitness_history)
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

    # Print best solutions for each algorithm using stored results
    for algo, res in results.items():
        # Print a concise best-solution table for each algorithm. This call
        # ensures both CSO and FPA results are printed separately rather than
        # only printing the last algorithm.
        print_best_solutions(res["runs"], algo, res.get("time_avg", avg_time), P_max=P_max)

    out_basename = os.path.join("output", f"population_summary_P{P_max}")
    print_population_summary_table(
        results["CSO"]["population_histories"], results["CSO"]["fitness_histories"],
        results["FPA"]["population_histories"], results["FPA"]["fitness_histories"],
        P_max, K, max_iter, num_snapshots=5, output_path=out_basename
    )
    # Optionally: to limit the number of snapshot columns in the printed table,
    # callers can pass `selected_iterations` or set `num_snapshots` (default 5)

    return results

def plot_mean_convergence(results, max_iter=100, P_max=None, save_path=None):
# /**
#  * @brief Plot mean best-so-far and population-average curves with ±1 std shading.
#  *
#  * Inputs:
#  *   - results (dict): aggregated results from evaluate_algorithms
#  *   - max_iter (int): plotting horizon
#  *   - P_max (float|None): displayed in title if provided
#  *   - save_path (str|None): path to save PNG; if None, show plot interactively
#  *
#  * Outputs:
#  *   - produces and optionally saves a convergence PNG; returns None.
#  */
    plt.figure(figsize=(8, 6))

    for algo_name, res in results.items():
        histories = res["histories"]
        fitness_histories = res["fitness_histories"]
        min_len = min(len(h) for h in histories)
        histories = [h[:min_len] for h in histories]
        mean_curve = np.mean(histories, axis=0)
        std_curve = np.std(histories, axis=0)
        # iteration axis used for plotting (length = length of mean_curve)
        iters = np.arange(len(mean_curve))
        # Sửa lỗi: avg_fitness shape (num_runs, min_len), lấy trung bình theo từng iteration
        # avg_fitness: list các mảng (num_cats, min_len) hoặc (min_len,)
        # Chuyển về shape (num_runs, min_len) bằng cách lấy trung bình theo cá thể nếu cần
        # Xử lý shape cho avg_fitness
        avg_fitness_runs = []
        for fh in fitness_histories:
            fh = np.array(fh[:min_len])
            # fh can be shape (min_len, num_cats) or (num_cats, min_len) or (min_len,)
            if fh.ndim == 2:
                if fh.shape[0] == min_len:
                    # per-iteration arrays along axis 0: (min_len, num_cats)
                    avg_per_iter = np.mean(fh, axis=1)
                elif fh.shape[1] == min_len:
                    # per-iteration arrays along axis 1: (num_cats, min_len)
                    avg_per_iter = np.mean(fh, axis=0)
                else:
                    # unexpected layout: collapse to 1D by mean
                    avg_per_iter = np.mean(fh, axis=0).flatten()
            elif fh.ndim == 1:
                # already per-iteration average
                avg_per_iter = fh.copy()
            elif fh.ndim == 3:
                # collapse leading dims and average per-iteration
                # assume last axis is iterations if matching min_len
                if fh.shape[-1] == min_len:
                    avg_per_iter = np.mean(fh, axis=tuple(range(fh.ndim - 1)))
                else:
                    avg_per_iter = np.mean(fh, axis=0).flatten()
            else:
                avg_per_iter = np.ravel(fh)[:min_len]

            # ensure length == min_len
            if avg_per_iter.shape[0] != min_len:
                # try to broadcast or truncate/pad
                avg_per_iter = np.resize(avg_per_iter, min_len)

            avg_fitness_runs.append(avg_per_iter)

        avg_fitness_arr = np.vstack(avg_fitness_runs) if len(avg_fitness_runs) > 0 else np.zeros((0, min_len))
        if avg_fitness_arr.size == 0:
            mean_avg_fitness = np.zeros(min_len, dtype=float)
        else:
            mean_avg_fitness = np.mean(avg_fitness_arr, axis=0)

        plt.plot(iters, mean_curve, label=f"{algo_name} Best (mean)", linewidth=2)
        plt.plot(iters, mean_avg_fitness, label=f"{algo_name} Avg (mean)", linestyle='--', linewidth=2)
        plt.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Sum Rate (bits/s/Hz)")
    title = "Convergence Comparison: CSO vs FPA (Best and Avg ± Std)"
    if P_max is not None:
        title += f" — P_max={P_max}"
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir and not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Convergence plot saved: {save_path}")
    else:
        plt.show()


def compute_convergence_metrics(results, threshold=0.95):
# /**
#  * @brief Compute convergence speed metrics across runs.
#  *
#  * Inputs:
#  *   - results (dict): aggregated results dict from evaluate_algorithms
#  *   - threshold (float): fraction of final best used to define convergence (default 0.95)
#  *
#  * Outputs:
#  *   - returns dict mapping algorithm name -> metrics (iters_needed_mean/median/std, auc_mean/std, final_mean)
#  */
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


def print_best_solutions(all_runs, algo_name, avg_time, P_max=None):
# /**
#  * @brief Print a small table of sampled best-so-far solutions per run.
#  *
#  * Inputs:
#  *   - all_runs (list): each element is an iterable/list of best-solution dicts for a run
#  *   - algo_name (str): algorithm label used in headers
#  *   - avg_time (float): average runtime per run (seconds)
#  *   - P_max (float|None): optional power cap for header
#  *
#  * Outputs:
#  *   - prints a sampled table to stdout (pandas DataFrame / tabulate) and returns None.
#  */
    # Header includes the P_max value when provided so printed outputs are
    # self-describing for multi-P experiments.
    header = f"\n=== {algo_name} Results (average time: {avg_time:.4f} s) ==="
    if P_max is not None:
        header = f"\n=== {algo_name} Results (P_max={P_max}) (average time: {avg_time:.4f} s) ==="
    print(header)
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
    # Ensure outputs go to the output directory
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run experiments for several P_max values and save outputs per value
    for P in [1.0, 10.0, 100.0]:
        print(f"\n--- Running experiments for P_max={P} ---")
        results = evaluate_algorithms(algorithms, H, K=K, P_max=P, num_runs=10, max_iter=100)
        plot_file = os.path.join(OUTPUT_DIR, f"convergence_P{P}.png")
        plot_mean_convergence(results, max_iter=100, P_max=P, save_path=plot_file)

        # Compute convergence metrics for this P and save to CSV
        metrics = compute_convergence_metrics(results, threshold=0.95)
        # Flatten metrics into table rows: algo, iters_mean, iters_median, iters_std, auc_mean, auc_std, final_mean, P_max
        rows = []
        for algo, m in metrics.items():
            rows.append({
                'P_max': P,
                'Algorithm': algo,
                'iters_needed_mean': m['iters_needed_mean'],
                'iters_needed_median': m['iters_needed_median'],
                'iters_needed_std': m['iters_needed_std'],
                'auc_mean': m['auc_mean'],
                'auc_std': m['auc_std'],
                'final_mean': m['final_mean']
            })
        metrics_df = pd.DataFrame(rows)
        metrics_csv = os.path.join(OUTPUT_DIR, f"convergence_metrics_P{P}.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"Convergence metrics saved: {metrics_csv}")

        # Print a short console summary including P_max
        print("\n=== Convergence speed metrics (reach 95% of final best) P_max={:.1f} ===".format(P))
        for algo, m in metrics.items():
            print(f"P={P:.1f} {algo}: mean iters={m['iters_needed_mean']:.1f}, median={m['iters_needed_median']:.1f}, std={m['iters_needed_std']:.1f}, AUC_mean={m['auc_mean']:.3f}")
        for algo, res in results.items():
            print(f"\nP={P:.1f} === {algo} ===")
            print(f"Best overall: {res['best_overall']:.6f}")
            print(f"Average best: {res['avg']:.6f}")
            print(f"Std dev: {res['std']:.6f}")
            print(f"Average time: {res['time_avg']:.4f}s")

        # Save population summary outputs into output directory
        out_basename = os.path.join(OUTPUT_DIR, f"population_summary_P{P}")
        print_population_summary_table(
            results["CSO"]["population_histories"], results["CSO"]["fitness_histories"],
            results["FPA"]["population_histories"], results["FPA"]["fitness_histories"],
            P, K, 100, num_snapshots=5, output_path=out_basename
        )