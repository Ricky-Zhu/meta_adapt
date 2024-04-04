from libero.lifelong.metric import *


def split_evaluation(cfg, algo, benchmark, task_ids):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        success_rate = evaluate_one_task_success(cfg, algo, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)


def split_evaluate_one_task_success(
        cfg, algo, task, task_emb, task_id, seen_initial_list, unseen_initial_list, eval_num, sim_states=None, task_str=""
):
    """
    Evaluate a single task's success rate
    sim_states: if not None, will keep track of all simulated states during
                evaluation, mainly for visualization and debugging purpose
    task_str:   the key to access sim_states dictionary
    """
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id)

        algo.eval()
        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # Try to handle the frame buffer issue
        env_creation = False

        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)

        def evaluate_on_given_init_list(init_list):
            num_success = 0
            for i in range(init_list):
                env.reset()

                init_states_ = init_states[i:i + env_num]

                dones = [False] * env_num
                steps = 0
                algo.reset()
                obs = env.set_init_state(init_states_)

                # dummy actions [env_num, 7] all zeros for initial physics simulation
                dummy = np.zeros((env_num, 7))
                for _ in range(5):
                    obs, _, _, _ = env.step(dummy)

                if task_str != "":
                    sim_state = env.get_sim_state()
                    for k in range(env_num):
                        if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                            sim_states[i * env_num + k].append(sim_state[k])

                while steps < cfg.eval.max_steps:
                    steps += 1

                    data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                    actions = algo.policy.get_action(data)

                    obs, reward, done, info = env.step(actions)

                    # record the sim states for replay purpose
                    if task_str != "":
                        sim_state = env.get_sim_state()
                        for k in range(env_num):
                            if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                                sim_states[i * env_num + k].append(sim_state[k])

                    # check whether succeed
                    for k in range(env_num):
                        dones[k] = dones[k] or done[k]

                    if all(dones):
                        break

                # a new form of success record
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval:
                        num_success += int(dones[k])

                env.close()
                gc.collect()
                return num_success / len(init_list)
        seen_suc_rate = 0.
        unseen_suc_rate = 0.
        for i in range(eval_num):
            seen_suc_rate += evaluate_on_given_init_list(seen_initial_list)
            unseen_suc_rate += evaluate_on_given_init_list(unseen_initial_list)
        seen_suc_rate = seen_suc_rate/eval_num
        unseen_suc_rate = unseen_suc_rate/eval_num
    print(f"[info] seen: {seen_suc_rate} unseen: {unseen_suc_rate}")
    return seen_suc_rate, unseen_suc_rate
