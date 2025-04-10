seed=$1
demo_num=$2
suite=LIBERO_GOAL
pre_trained_model_path=../scripts/experiments/LIBERO_GOAL/PreTrainMultitask/BCTransformerPolicy_seed10000/run_001/multitask_model_ep10.pth
policy_type=LoraBCTPolicy
python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[5] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[5] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[6] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[6] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[7] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[7] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[8] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[8] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[9] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[9] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type


pre_trained_model_path=../scripts/experiments/LIBERO_GOAL/PreTrainMultitask/BCViLTPolicy_seed10000/run_007/multitask_model_ep20.pth
policy_type=LoraBCViLTPolicy
python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[5] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[5] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[6] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[6] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[7] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[7] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[8] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[8] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[9] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[9] adaption_suite=$suite batch_size=32 pre_trained_model_path=$pre_trained_model_path policy_type=$policy_type

