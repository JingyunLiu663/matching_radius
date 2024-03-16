from simulator_env import Simulator
import numpy as np
from config import *
from path import *
import warnings
import torch
warnings.filterwarnings("ignore")
import logging
from utilities import *
import argparse
import train_test
from datetime import datetime
from path import *

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-radius', type=float, default=1.0, help="for radius method = 'fixed', max pickup radius")
    # for RL 
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
    # for RL
    parser.add_argument('-radius_method', type=str, default="fixed", help='rl/fixed/greedy') 
    parser.add_argument('-experiment_mode', type=str, default="train", help="train/test")
    args = parser.parse_args()
    return args

def is_gpu_available():
    if torch.cuda.is_available():
        logging.info("GPU is available")

def get_logger(logger_name, log_file, formatter_pattern=None, console=False):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create the directory for logging
    os.makedirs(logger_name, exist_ok=True)
    # attach a timestamp to the log file, write the log file to the directory
    file_handler = logging.FileHandler(os.path.join(logger_name, f"{datetime.today()}_{log_file}.log"))
    file_handler.setLevel(logging.INFO)
    if formatter_pattern:
        formatter = logging.Formatter(formatter_pattern)
        file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        if formatter_pattern:
            console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

if __name__ == "__main__":
    # Check whether GPU is available
    is_gpu_available()
    # Get command line arguments
    args = get_args()
    # Initialize metrics logger
    
    if args.radius_method == 'rl':
        loss_logger = get_logger(logger_name='model_loss', log_file=f'{args.rl_agent}_loss')
        logger = get_logger(logger_name='metrics_logger', log_file=f'{args.rl_agent}_{args.experiment_mode}', formatter_pattern='%(asctime)s - %(message)s', console=True)
    elif args.radius_method == 'fixed':
        logger = get_logger(logger_name='metrics_logger', log_file=f'{args.radius_method}_r{args.radius}', formatter_pattern='%(asctime)s - %(message)s', console=True)

    # Initialize the simulator 
    simulator = Simulator(**env_params)
    # Adjust the simulator environment based on command line arguments
    train_test.update_simulator_args(simulator, args)

    if simulator.radius_method == "rl" and simulator.experiment_mode == 'train':
        logging.info("RL training process:")
        agent = train_test.create_agent(args)
        # Load pre-trained model if necessary
        if env_params['pre_trained']:
            train_test.load_pretrained_model(agent, args, env_params, model_epoch=100)
        # Run simulation - train mode
        train_test.simulation_train(args, agent, simulator, logger, loss_logger)
        # Close TensorBoard writer 
        agent.train_writer.close()

    elif simulator.radius_method == "rl" and simulator.experiment_mode == 'test':
        logging.info("RL testing process:")
        # Initialize the RL agent for matching radius (dqn, double dqn, etc...)
        agent = train_test.create_agent(args)
        # Load pre-trained model for validation & testing
        train_test.load_pretrained_model(agent, args, env_params,model_epoch=100)  
        # Run simulation - test mode   
        testing.simulation_test(args, agent, simulator, logger, test_num=10)
        # Close TensorBoard writer
        agent.test_writer.close()

    elif simulator.radius_method == "fixed":
        logging.info(f"fixed radius = {simulator.maximal_pickup_distance}")
        # Initialize the RL agent for matching radius (dqn, double dqn, etc...)
        train_test.simulation_fixed(simulator, logger, test_num=10)
        
    elif simulator.radius_method == "greedy":
        logging.info(f"greedy radius")
        