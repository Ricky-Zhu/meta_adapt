import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark, get_benchmark_dict
from libero.libero import get_libero_path, set_libero_default_path

benchmark_instance = get_benchmark_dict()['libero_object']()


bddl_files_default_path = get_libero_path("bddl_files")

task_id = 9
task = benchmark_instance.get_task(task_id)

env_args = {
    "bddl_file_name": os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file),
    "camera_heights": 128,
    "camera_widths": 128
}

env = OffScreenRenderEnv(**env_args)
init_states = benchmark_instance.get_task_init_states(task_id)

env.seed(0)
env.reset()
env.set_init_state(init_states[0])
for _ in range(5):
        obs, _, _, _ = env.step([0.] * 7)