seed=$1
demo_num=$2
suite=LIBERO_GOAL
pre_trained_model_path=../scripts/experiments/LIBERO_GOAL/PreTrainMultitask/BCTransformerPolicy_seed10000/run_001/multitask_model_ep10.pth
policy_type=LoraBCTPolicy

python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[5] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
