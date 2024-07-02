#demo_num=$(( $1 ))
#seed=$(( $2 ))
path=../scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_003/multitask_model.pth
seed=$(( $1 ))

#python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=5 seed=$seed pre_trained_model_path=$path
#sleep 2
#python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=5 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=5 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=5 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=5 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=5 seed=$seed pre_trained_model_path=$path


#python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=6 seed=$seed pre_trained_model_path=$path
#sleep 2
#python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=6 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=6 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=6 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=6 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=6 seed=$seed pre_trained_model_path=$path


#python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=7 seed=$seed pre_trained_model_path=$path
#sleep 2
#python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=7 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=7 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=7 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=7 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=7 seed=$seed pre_trained_model_path=$path


#python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=8 seed=$seed pre_trained_model_path=$path
#sleep 2
#python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=8 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=8 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=8 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=8 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=8 seed=$seed pre_trained_model_path=$path


#python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=9 seed=$seed pre_trained_model_path=$path
#sleep 2
#python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=9 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=9 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=9 seed=$seed pre_trained_model_path=$path

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=9 seed=$seed pre_trained_model_path=$path
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=9 seed=$seed pre_trained_model_path=$path
