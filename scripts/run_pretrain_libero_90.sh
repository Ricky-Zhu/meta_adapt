python ../pre_training/trainer.py task_creation.task_suite="libero_90" benchmark_name="LIBERO_90" train.batch_size=64
sleep 2
python ../pre_training/evaluate.py task_creation.task_suite="libero_90" benchmark_name="LIBERO_90" train.batch_size=64