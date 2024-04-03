# pre-train the model
```
cd scripts
python ../pre_training/trainer.py task_creation.pre_training_num=5
```

the best pre-trained model will be saved into `scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_00*`

# adaptation

`python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=10 adaptation.adaptation_task_id=9`