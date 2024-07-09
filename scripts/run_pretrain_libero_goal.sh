policy=$(( $1 ))
python ../pre_training/trainer.py task_creation.task_suite="libero_goal" task_creation.select_tasks=[0,1,2,3,4] benchmark_name="LIBERO_GOAL" train.batch_size=64 policy=$policy
sleep 2
python ../pre_training/evaluate.py task_creation.task_suite="libero_goal" task_creation.select_tasks=[0,1,2,3,4] benchmark_name="LIBERO_GOAL" train.batch_size=64 policy=$policy