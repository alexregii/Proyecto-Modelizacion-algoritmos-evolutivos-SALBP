import numpy as np

def simulate_chain(n, m, p, num_runs=1000):
    counts = np.zeros(m)
    for _ in range(num_runs):
        pos = 0
        sequence = []
        for _ in range(n):
            sequence.append(pos)
            if pos < m - 1 and np.random.rand() < p:
                pos += 1
        for s in sequence:
            counts[s] += 1
    probs = counts / (n * num_runs)
    return probs

def find_best_p(n, m, num_runs=1000):
    best_p = None
    best_score = float('inf')
    best_dist = None
    for p in np.linspace(0.01, 0.99, 200):
        dist = simulate_chain(n, m, p, num_runs)
        uniform = np.ones(m) / m
        score = np.sum((dist - uniform) ** 2)  # error cuadrático medio
        if score < best_score:
            best_score = score
            best_p = p
            best_dist = dist
    return best_p, best_dist

p_opt, dist = find_best_p(70, 10, num_runs=1000)
print(p_opt)
print("Distribución final:", np.round(dist, 4))