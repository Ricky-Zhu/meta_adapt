seed=$1

#./run_online_adaptation_libero_spatial.sh $seed 5
./run_online_adaptation_libero_object.sh $seed 5
./run_online_adaptation_libero_goal.sh $seed 5

./run_online_adaptation_libero_spatial.sh $seed 10
#./run_online_adaptation_libero_object.sh $seed 10
./run_online_adaptation_libero_goal.sh $seed 10


./run_online_adaptation_libero_spatial.sh $seed 20
./run_online_adaptation_libero_object.sh $seed 20
./run_online_adaptation_libero_goal.sh $seed 20