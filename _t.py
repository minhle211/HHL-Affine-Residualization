import sys, time
sys.path.insert(0, "bench"); sys.path.insert(0, ".")
from modern_compare import run_system
for N in (256, 1024):
    t = time.time()
    run_system(("wishart", N, 1e4, 1e-3, 0, (0.01, 0.1, 1.0), 0, 0.125, 0.5, 16))
    t1 = time.time()
    run_system(("wishart", N, 1e4, 1e-3, 1, (0.01, 0.1, 1.0), 0, 0.125, 0.5, 16))
    print(N, t1 - t, time.time() - t1)
