# DyPs
This is the code for paper "Hierarchical Multi-Agent Reinforcement Learning for Spatio-Temporal Mobile Resource Allocation"


## Installation and Setups 
``` python
conda create -n DyPs python=3.9
conda activate CoTa
pip install -r requirements.txt
```


## Run Experiments
* Command to run our method

``` python
cd run
python run_lstm_hrl.py
```

* Command for baselines

```python
cd run
python run_hrl.py   # Our method without lstm
python run_cvae.py  # Our method withour hierarchical structure
python PS_noid.py   # Naive parameter sharing 
```

## Visualizations
* You can visualize the learning curves by tensorboard.
``` python
tensorboard --logdir logs
```

* You can visualize the city map and demand-supply heat map by following jupyters.
``` python
plot/grid_map.ipynb
```

* training curve of ride-hailing (2) scenario

![曲线图](training_curve.pdf)