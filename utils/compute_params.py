import os
from glob import glob

path = '../scripts/experiments/lora_adaptation'
for task in os.listdir(path):
    task_path = os.path.join(path, task)
    for demo in os.listdir(task_path):
        demo_path = os.path.join(task_path, demo)
        for seed in os.listdir(demo_path):
            seed_path = os.path.join(demo_path, seed)
            try:
                with open(os.path.join(seed_path, 'performance.txt'), 'r') as f:
                    print(f.read())
            except:
                print(f'{seed_path} has no performance txt\n')
