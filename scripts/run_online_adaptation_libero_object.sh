seed=$1
demo_num=$2
pre_trained_model_path=../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth
policy_type=LoraBCTPolicy
python ../online_adapt/online_continual_adapt.py adapt_demo_num_each_task=$demo_num seed=$seed pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../online_adapt/online_adapt_evaluation.py adapt_demo_num_each_task=$demo_num seed=$seed pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type


pre_trained_model_path=../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCViLTPolicy_seed10000/run_018/multitask_model_ep50.pth
policy_type=LoraBCViLTPolicy
python ../online_adapt/online_continual_adapt.py adapt_demo_num_each_task=$demo_num seed=$seed pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../online_adapt/online_adapt_evaluation.py adapt_demo_num_each_task=$demo_num seed=$seed pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type