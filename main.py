from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
import os
from utilities import *
from dqn import DqnAgent
import config
from matplotlib import pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-action_space', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0], help='action space - list of matching radius')

    parser.add_argument('-num_layers', type=int, default=2, help='Number of fully connected layers')
    parser.add_argument('-layers_dimension_list', type=int, nargs='+', default=[256, 256], help='List of dimensions for each layer')
    parser.add_argument('-lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.9, help='discount rate gamma')
    parser.add_argument('-epsilon', type=float, default=0.9, help='epsilon greedy - begin epsilon')
    parser.add_argument('-eps_min', type=float, default=0.01, help='epsilon greedy - end epsilon')
    parser.add_argument('-eps_dec', type=float, default=0.997, help='epsilon greedy - epsilon decay per step')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    training_log = {
        'epoch_running_times': [],
        'epoch_total_rewards': [],
        'epoch_total_orders': [],
        'epoch_matched_orders': []
    }

    # initialize the simulator
    simulator = Simulator(**env_params)

    if env_params['rl_mode'] == "matching_radius":
        if simulator.experiment_mode == 'train':
            print("training process:")
            # initialize the RL agent for matching radius setting
            # initialize the RL agent for matching radius setting
            agent = DqnAgent(action_space=args.action_space, num_layers=args.num_layers, layers_dimension_list=args.layers_dimension_list, lr=0.005,
                             gamma=0.9, epsilon=0.9, eps_min=0.01, eps_dec=0.997, target_replace_iter=100, batch_size=8,
                             mem_size=2000)
            parameter_path = f"/pre_trained/{env_params['rl_agent']}/{env_params['rl_agent']}_{'_'.join(map(str, args.layers_dimension_list))}_model.pt"
            # use pre-trained model
            if env_params['pre_trained']:
                agent.load_parameters(parameter_path)

            # log: keep track of the total reward
            total_reward_record = np.zeros(NUM_EPOCH)
            for epoch in range(NUM_EPOCH):
                simulator.experiment_date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
                # initialize the environment
                #   simulator.driver_table is constructed (a deep copy of sampled simulator.driver_info)
                simulator.reset()
                start_time = time.time()
                # for every time interval do:
                for step in range(simulator.finish_run_step):
                    driver_table = deepcopy(simulator.driver_table)
                    idle_driver_table = driver_table[
                        (simulator.driver_table['status'] == 0) | (simulator.driver_table['status'] == 4)]
                    # print("idle_driver_table's shape: ", idle_driver_table.shape)
                    # Collect the action taken by each driver, so that we can run the dispatch algorithm and update
                    # the simulating environment
                    for index, row in idle_driver_table.iterrows():
                        # map state to action for each driver
                        state = (simulator.time, row['grid_id'])
                        action_index = agent.choose_action(state)
                        # keep track of the action for each driver
                        simulator.driver_table['action_index'] = action_index
                        simulator.driver_table['matching_radius'] = args.action_space[action_index]
                    # observe the transition and store the transition in the replay buffer
                    transition_buffer = simulator.step(idle_driver_table)
                    if transition_buffer:
                        states, action_indices, rewards, next_states = transition_buffer
                        agent.store_transition(states, action_indices, rewards, next_states)
                    # FIXME: feed the transition to the DQN after certain batch of data is collected
                    if agent.transition_count % UPDATE_INTERVAL == 0:
                        agent.learn()
                end_time = time.time()
                total_reward_record[epoch] = simulator.total_reward

                print('epoch:', epoch)
                print('epoch running time: ', end_time - start_time)
                print('epoch total reward: ', simulator.total_reward)
                print("total orders", simulator.total_request_num)
                print("matched orders", simulator.matched_requests_num)
                # print("loss", agent.loss_values)

                training_log['epoch_running_times'].append(end_time - start_time)
                training_log['epoch_total_rewards'].append(simulator.total_reward)
                training_log['epoch_total_orders'].append(simulator.total_request_num)
                training_log['epoch_matched_orders'].append(simulator.matched_requests_num)

                # with open(f"output/{env_params['rl_agent']}.pickle", "wb") as f:
                #     pickle.dump(simulator.record, f)
                # if epoch % 200 == 0:  # save the result every 200 epochs
                #     agent.save_parameters(epoch)
                if epoch % 5 == 0:
                    with open(
                            f"./training_log/{env_params['rl_agent']}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}.pickle",
                            'wb') as f:
                        pickle.dump(training_log, f)
                    with open(
                            f"./training_log/{env_params['rl_agent']}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_losses.pickle",
                            "wb") as f:
                        pickle.dump(agent.loss_values, f)
                    agent.save_parameters(parameter_path)

            with open(
                    f"./training_log/{env_params['rl_agent']}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}.pickle",
                    'wb') as f:
                pickle.dump(training_log, f)
            with open(
                    f"./training_log/{env_params['rl_agent']}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_losses.pickle",
                    'wb') as f:
                pickle.dump(agent.loss_values, f)