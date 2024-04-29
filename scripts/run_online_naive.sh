seed=$(( $1 ))

python ../online_adapt/online_continual_adapt.py meta_support_num=0 meta_query_num=0 seed=$seed
sleep 2
python ../online_adapt/online_adapt_evaluation.py meta_support_num=0 meta_query_num=0 seed=$seed

