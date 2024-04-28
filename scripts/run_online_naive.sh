support_num=$(( $1 ))
query_num=$(( $2 ))


python ../online_adapt/online_continual_adapt.py meta_support_num=$support_num meta_query_num=$query_num
sleep 2
python ../online_adapt/online_adapt_evaluation.py meta_support_num=$support_num meta_query_num=$query_num

