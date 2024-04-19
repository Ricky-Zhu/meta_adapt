
python ../adaptation/online_continual_adapt.py adapt_demo_num_each_task=10 meta_update_epochs=50
python ../adaptation/online_continual_adapt.py adapt_demo_num_each_task=10 meta_update_epochs=150
python ../adaptation/online_continual_adapt.py adapt_demo_num_each_task=10 meta_update_epochs=200

python ../adaptation/online_continual_adapt.py adapt_demo_num_each_task=10 meta_update_epochs=150 meta_support_num=10
python ../adaptation/online_continual_adapt.py adapt_demo_num_each_task=10 meta_update_epochs=150 meta_support_num=15
python ../adaptation/online_continual_adapt.py adapt_demo_num_each_task=10 meta_update_epochs=150 meta_support_num=10 meta_query_num=20