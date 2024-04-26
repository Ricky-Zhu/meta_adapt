demo_num=$(( $1 ))
seed=$(( $2 ))

python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=5 adaptation.seed=$seed
