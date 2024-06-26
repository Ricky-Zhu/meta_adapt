import numpy as np
import os

path = "../scripts/experiments/lora_adaptation/LoraBCViLTPolicy"

summary = {}

for task in os.listdir(path):
    summary[task] = {}

    for demo_num in os.listdir(os.path.join(path, task)):
        summary[task][demo_num] = []
        for seed in os.listdir(os.path.join(path, task, demo_num)):
            performance_path = os.path.join(path, task, demo_num, seed, 'performance.txt')
            with open(performance_path, 'r') as f:
                cont = f.read()
                f.close()
            score = cont.split(']')[0].split('[')[-1]
            summary[task][demo_num].append(float(score))

        avg = np.mean(summary[task][demo_num])
        std = np.std(summary[task][demo_num])
        summary[task][demo_num] = [avg, std]

print(summary)

demo_num_keys = list(summary[list(summary.keys())[0]].keys())
for demo_num in demo_num_keys:
    demo_suc = []
    for task in summary.keys():
        for k_demo in summary[task].keys():
            if k_demo == demo_num:
                demo_suc.append(summary[task][k_demo][0])

    print(f'{demo_num}: {np.mean(demo_suc)}')
