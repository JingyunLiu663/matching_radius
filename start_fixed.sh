#!/bin/bash

# get the rl_agent, adjust, and experiment_mode from the user
read -p "Enter the Experiment Mode (train or test): " experiment_mode
read -p "Enter the RL Mode: " radius_method
read -p "Enter the maximum radius: " radius
read -p "Enter the cruise flag: " cruise

# create the tmux session
radius_name=${radius//./_}
session_name="${radius_method}_${experiment_mode}_${radius_name}_cruise${cruise}"
tmux new-session -d -s ${session_name}

# send the conda activate command to the tmux session
tmux send-keys -t ${session_name} "conda activate simulator" C-m

# send the appropriate python command to the tmux session based on adjust and experiment_mode
tmux send-keys -t ${session_name} "python main.py -radius_method ${radius_method}  -experiment_mode ${experiment_mode} -radius ${radius} -cruise ${cruise}" C-m