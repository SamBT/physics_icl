import sys
import os

models = [f.split("/")[-1] for f in sys.argv[1:]]
n_per = 10

n_curr = 0
while n_curr < len(models):
    args = " ".join(models[n_curr:n_curr+n_per])
    os.system(f"sbatch evaluation.sh {args}")
    n_curr += n_per