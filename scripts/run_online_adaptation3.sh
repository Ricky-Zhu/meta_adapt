support_num=$(( $1 ))
query_num=$(( $2 ))


python ../adaptation/online_continual_adapt.py adapt_demo_num_each_task=10 meta_update_epochs=50 meta_support_num=$support_num meta_query_num=$query_num
sleep 2
python ../adaptation/online_adapt_evaluation.py adapt_demo_num_each_task=10 meta_update_epochs=50 meta_support_num=$support_num meta_query_num=$query_num

