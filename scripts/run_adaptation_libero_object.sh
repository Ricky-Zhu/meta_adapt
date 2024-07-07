seed=$(( $1 ))
demo_num=$(( $2 ))

python ../adaptation/adapt_train.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[1,2,3,4,5,6,7,8] adaption_suite=LIBERO_OBJECT batch_size=32
sleep 2
python ../adaptation/evaluation_single.py adapt_demo_num_each_task=$demo_num seed=$seed adapt_all=false adaptation_tasks=[1,2,3,4,5,6,7,8] adaption_suite=LIBERO_OBJECT batch_size=32