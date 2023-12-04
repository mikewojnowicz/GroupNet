# GroupNet: Application to HSRDM

This allows application of GroupNet to the basketball experiment from the HSRDM paper.

For training, run a command like
```
python src/groupnet/compare/train.py --num_train_games 5 --num_epochs 10 --its_per_epoch 1000
```

For forecasts, run a command like

```
python src/groupnet/compare/forecast.py --num_train_games 5 --model_name 10 
#usually use the model saved from the last epoch.
```