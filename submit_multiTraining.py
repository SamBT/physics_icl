import os
import sys

directory = sys.argv[1]
num_per_job = int(sys.argv[2])

mem_per_yaml = 2
all_yamls = [f for f in os.listdir(directory) if ".yaml" in f]

i = 0
if len(all_yamls) > num_per_job:
    num_per_job = len(all_yamls)
while i < len(all_yamls):
    yamls = all_yamls[i:i+num_per_job]
    append = f"--mem={int(len(yamls)*mem_per_yaml)}G -c {len(yamls)}"
    files = [f"{directory}/{f}" for f in yamls]
    submit = f"sbatch {append} submit.sh {' '.join(files)}"
    os.system(submit)
    #print(submit)
    i += num_per_job