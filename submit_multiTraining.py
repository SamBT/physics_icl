import os
import sys
from pathlib import Path

num_per_job = int(sys.argv[1])
target = sys.argv[2:] #inputs, could be list of yamls or directory containing yamls
mem_per_yaml = 2 # GB of RAM to allocate per yaml

if len(target) == 1 and os.path.isdir(target):
    all_yamls = [f for f in os.listdir(target[0]) if ".yaml" in f and "default" not in f]
    directories = len(all_yamls)*[target]
else:
    all_yamls = [Path(t).name for t in target]
    directories = [Path(t).parents[0] for t in target]

i = 0
while i < len(all_yamls):
    yamls = all_yamls[i:i+num_per_job]
    append = f"--mem={int(len(yamls)*mem_per_yaml)}G -c {len(yamls)}"
    files = [f"{directories[ii]}/{f}" for ii,f in enumerate(yamls)]
    submit = f"sbatch {append} submit.sh {' '.join(files)}"
    os.system(submit)
    #print(submit)
    i += num_per_job