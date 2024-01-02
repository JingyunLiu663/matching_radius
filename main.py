from simulator_env import Simulator
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
import torch
warnings.filterwarnings("ignore")
import logging
from utilities import *
import argparse
from log_generation import ModelTracker, EpochPerformanceTracker
import training, testing

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-radius', type=float, default=1.0, help="for radius method = 'fixed', max pickup radius")
    parser.add_argument('-rl_agent', type=str, default="dqn", help='RL agent') 
    parser.add_argument('-action_space', type=float, nargs='+',
                        default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
                        help='action space - list of matching radius')
    parser.add_argument('-num_layers', type=int, default=2, help='Number of fully connected layers')
    parser.add_argument('-dim_list', type=int, nargs='+', default=[128, 128],
                        help='List of dimensions for each layer')
    parser.add_argument('-lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.99, help='discount rate')
    # thw epsilon will reach close to eps_min after about 2000  - 1.0 * 0.9978^{2000} = 0.01
    # epsilon_dec = (desired_epsilon / epsilon_start) ** (1/steps)
    parser.add_argument('-epsilon', type=float, default=1.0, help='epsilon greedy - begin epsilon')
    parser.add_argument('-eps_min', type=float, default=0.01, help='epsilon greedy - end epsilon')
    parser.add_argument('-eps_dec', type=float, default=0.9978, help='epsilon greedy - epsilon decay per step')
    parser.add_argument('-target_update', type=int, default=2000, help='update frequency for target network')

    parser.add_argument('-adjust_reward', type=int, default=0, 
                        help='apply immediate reward(0), or adjust the reward by matching radius(1)') 
    parser.add_argument('-radius_method', type=str, default="rl", help='rl/fixed/greedy') 
    parser.add_argument('-experiment_mode', type=str, default="train", help="train/test")
    parser.add_argument('-cruise', type=int, default=1, help="cruise flag")
    args = parser.parse_args()
    return args

def is_gpu_available():
    if torch.cuda.is_available():
        logging.info("GPU is available")

if __name__ == "__main__":
    # Check whether GPU is available
    is_gpu_available()
    # Get command line arguments
    args = get_args()
    # Initialize the simulator 
    simulator = Simulator(**env_params)
    # Adjust the simulator environment based on command line arguments
    training.adjust_simulator(simulator, args)

    if simulator.radius_method == "rl" and simulator.experiment_mode == 'train':
        logging.info("RL training process:")
        # Initialize model tracker to store the RL agent parameters 
        model_tracker = ModelTracker()
        # Create the RL agent for matching radius (dqn, double dqn, etc...)
        agent = training.create_agent(args)
        # Load pre-trained model if necessary
        training.load_pretrained_model(agent, args, env_params)
        # Run simulation - train mode
        training.simulation_train(args, agent, simulator, model_tracker)
        # Close TensorBoard writer 
        agent.train_writer.close()

    elif simulator.radius_method == "rl" and simulator.experiment_mode == 'test':
        logging.info("RL testing process:")
        # Initialize the RL agent for matching radius (dqn, double dqn, etc...)
        agent = training.create_agent(args)
        # Load pre-trained model for validation & testing
        training.load_pretrained_model(agent, args, env_params,model_epoch=7)  
        # Run simulation - test mode   
        testing.simulation_test(args, agent, simulator, test_num=10)
        # Close TensorBoard writer
        agent.test_writer.close()

    elif simulator.radius_method == "fixed":
        logging.info(f"fixed radius = {simulator.maximal_pickup_distance}")
        # Initialize the RL agent for matching radius (dqn, double dqn, etc...)
        training.simulation_fixed(simulator, test_num=10)
        
    elif simulator.radius_method == "greedy":
        logging.info(f"greedy radius")
        