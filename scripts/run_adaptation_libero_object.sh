seed=$(( $1 ))

python ../adaptation/adapt_train.py adapt_demo_num_each_task=5 seed=$seed adapt_all=false adaptation_tasks=[1,3] batch_size=16
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=5 seed=$seed adapt_all=false adaptation_tasks=[1,3] batch_size=16