#demo_num=$(( $1 ))
#seed=$(( $2 ))

seed=$(( $1 ))

python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=5 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=5 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=5 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=5 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=5 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=5 seed=$seed


python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=6 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=6 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=6 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=6 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=6 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=6 seed=$seed


python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=7 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=7 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=7 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=7 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=7 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=7 seed=$seed


python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=8 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=8 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=8 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=8 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=8 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=8 seed=$seed


python ../adaptation/adapt_train.py adapt_demo_num_each_task=1 adaptation_task_id=9 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=1 adaptation_task_id=9 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 adaptation_task_id=9 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 adaptation_task_id=9 seed=$seed

python ../adaptation/adapt_train.py adapt_demo_num_each_task=10 adaptation_task_id=9 seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=10 adaptation_task_id=9 seed=$seed
