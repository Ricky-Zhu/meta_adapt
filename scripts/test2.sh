model_path='../scripts/experiments/LIBERO_90/PreTrainMultitask/BCTransformerPolicy_seed10000/run_001/multitask_model_ep20.pth'
policy='LoraBCTPolicy'

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adapt_all=false adaptation_tasks=[1,2,3,4,5,6,7,8] adaption_suite=LIBERO_SPATIAL batch_size=64 pre_trained_model_path=$model_path policy_type=$policy
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adapt_all=false adaptation_tasks=[1,2,3,4,5,6,7,8] adaption_suite=LIBERO_SPATIAL batch_size=64 pre_trained_model_path=$model_path policy_type=$policy