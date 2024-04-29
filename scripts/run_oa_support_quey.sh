

python ../online_adapt/online_continual_adapt.py -m adapt_demo_num_each_task=10 meta_update_epochs=50 meta_support_num=10 meta_query_num=15 random_meta=false seed=0,100,10000
sleep 2
python ../online_adapt/online_adapt_evaluation.py -m adapt_demo_num_each_task=10 meta_update_epochs=50 meta_support_num=10 meta_query_num=15 random_meta=false seed=0,100,10000

python ../online_adapt/online_continual_adapt.py -m adapt_demo_num_each_task=10 meta_update_epochs=50 meta_support_num=10 meta_query_num=15 random_meta=true seed=0,100,10000
sleep 2
python ../online_adapt/online_adapt_evaluation.py -m adapt_demo_num_each_task=10 meta_update_epochs=50 meta_support_num=10 meta_query_num=15 random_meta=true seed=0,100,10000

