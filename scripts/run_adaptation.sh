demo_num=10
seed=100

python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=5 adaptation.seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=5 adaptation.seed=$seed

python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=6 adaptation.seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=6 adaptation.seed=$seed

python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=7 adaptation.seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=7 adaptation.seed=$seed

python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=8 adaptation.seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=8 adaptation.seed=$seed

python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=9 adaptation.seed=$seed
sleep 2
python ../adaptation/evaluation_single.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=9 adaptation.seed=$seed

#python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=6
#python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=7
#python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=8
#python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=9