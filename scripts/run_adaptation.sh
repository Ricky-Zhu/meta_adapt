demo_num=$1

python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=5
python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=6
python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=7
python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=8
python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=$demo_num adaptation.adaptation_task_id=9