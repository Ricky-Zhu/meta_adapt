# pre-train the model

```
cd scripts
python ../pre_training/trainer.py task_creation.pre_training_num=5
```

the best pre-trained model will be saved
into `scripts/experiments/LIBERO_OBJECT/PreTrainMultitask/BCTransformerPolicy_seed10000/run_00*`

# adaptation

The configs for adaptation are stored in configs/adaptation.yaml (the adaptation part other configs are inherited from
the pre-training exp or modified during the script)

<mark>The random seed in the post evaluation should be the same as that in the adaptation exp. This is because the lora
will initialize a fixed matrix which is not saved.</mark>

`python ../adaptation/adapt_train.py adaptation.adapt_demo_num_each_task=10 adaptation.adaptation_task_id=9`

# TODO List

- [ ] build more general pipeline for different task setup (goal, spatial, long)
- [ ] find out how to better leverage the benefits of task description (e.g. using CLIP )
