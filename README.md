# PolarDR: Polarity-aware Preference Trajectory Modeling for Effective Dynamic Recommendation
<img width="3951" height="1236" alt="PolarDR_Framework" src="https://github.com/user-attachments/assets/1e6958fe-463b-41a6-a3bf-3b9d85256e2a" />


This repository provides the official implementation of PolarDR, as described in the paper "PolarDR: Polarity-aware Preference Trajectory Modeling for Effective Dynamic Recommendation," published at the ACM International Conference on Web Search and Data Mining (WSDM 2027) as a full paper.

## Inputs
The structure of the input dataset is the following: ```| user_id | item_id | rating | timestamp |```.

The dataset is preprocessed through ```libray_data.py```.

## Outputs
The best model is saved in the ```best_models``` folder, and its accuracy is recorded in the ```log``` folder.

## Usage
To run PolarDR with TraceRec as the backbone on different datasets, use the following commands:
+ For Beauty:
```
python polardr.py --dataset Beauty --baseline tracerec --num_path_u 25 --num_path_i 15 --num_path_u_sample 20 --num_path_i_sample 20 --seed 0 --gpu 0 --project --orth_loss_weight 0.0001
```

+ For Digital Music:
```
python polardr.py --dataset Digital_Music --baseline tracerec --num_path_u 5 --num_path_i 10 --num_path_u_sample 100 --num_path_i_sample 100 --seed 0 --gpu 0 --project --orth_loss_weight 0.001
```

+ For Instant Video:
```
python polardr.py --dataset Instant_Video --baseline tracerec --num_path_u 35 --num_path_i 50 --num_path_u_sample 100 --num_path_i_sample 100 --seed 0 --gpu 0 --project --orth_loss_weight 0.001
```

+ For Video Games:
```
python polardr.py --dataset Video_Games --baseline tracerec --num_path_u 5 --num_path_i 15 --num_path_u_sample 120 --num_path_i_sample 120 --seed 0 --gpu 0 --project --orth_loss_weight 0.00001
```

+ For Yelp:
```
python polardr.py --dataset yelp --baseline tracerec --num_path_u 10 --num_path_i 30 --num_path_u_sample 70 --num_path_i_sample 70 --seed 0 --gpu 0 --project --orth_loss_weight 0.00001
```
