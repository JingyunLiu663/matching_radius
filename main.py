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
from a2c import A2CAgent
import config
from matplotlib import pyplot as plt
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-rl_agent', type=str, default="dqn", help='RL agent')
    # "dqn" "a2c"
    parser.add_argument('-action_space', type=float, nargs='+',
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
                        help='action space - list of matching radius')

    parser.add_argument('-num_layers', type=int, default=2, help='Number of fully connected layers')
    parser.add_argument('-layers_dimension_list', type=int, nargs='+', default=[256, 256],
                        help='List of dimensions for each layer')
    parser.add_argument('-lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('-gamma', type=float, default=0.99, help='discount rate gamma')

    # thw epsilon will reach close to eps_min after about 2000  - 0.9 * 0.9987^{2000} = 0.01
    # epsilon_dec = (desired_epsilon / epsilon_start) ** (1/steps)
    parser.add_argument('-epsilon', type=float, default=0.9, help='epsilon greedy - begin epsilon')
    parser.add_argument('-eps_min', type=float, default=0.01, help='epsilon greedy - end epsilon')
    parser.add_argument('-eps_dec', type=float, default=0.9987, help='epsilon greedy - epsilon decay per step')

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
            if args.rl_agent == "dqn":
                agent = DqnAgent(args.action_space, args.num_layers, args.layers_dimension_list, args.lr, args.gamma,
                                 args.epsilon, args.eps_min, args.eps_dec, target_replace_iter=10)
                parameter_path = (f"pre_trained/{args.rl_agent}/"
                                  f"{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_model.pth")
                print("dqn agent is created")
            elif args.rl_agent == "a2c":
                print("a2c agent created")
                agent = A2CAgent(policy_hidden_dim=128, value_hidden_dim=128, action_dim=len(args.action_space),
                                 learning_rate=args.lr)
                parameter_path = f"pre_trained/{args.rl_agent}/{args.rl_agent}_model.pth"
                print("a2c agent is created")
            # use pre-trained model
            else:
                pass
            if env_params['pre_trained']:
                agent.load_parameters(parameter_path)

            # log: keep track of the total reward
            total_reward_record = np.zeros(NUM_EPOCH)
            print(args.action_space)
            for epoch in range(NUM_EPOCH):
                simulator.experiment_date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
                # initialize the environment
                #   simulator.driver_table is constructed (a deep copy of sampled simulator.driver_info)
                simulator.reset()
                start_time = time.time()
                # for every time interval do:
                for step in range(simulator.finish_run_step):
                    # Get the boolean mask for idle drivers
                    is_idle = (simulator.driver_table['status'] == 0) | (simulator.driver_table['status'] == 4)
                    # Fetch grid_ids for the idle drivers
                    grid_ids = simulator.driver_table.loc[is_idle, 'grid_id'].values
                    time_slices = np.full_like(grid_ids, simulator.time)
                    # Determine the action_indices for the idle drivers
                    action_indices = agent.choose_action((time_slices, grid_ids))
                    # Calculate matching radius for the idle drivers
                    action_space_array = np.array(args.action_space)
                    matching_radii = action_space_array[action_indices]
                    # Update the simulator.driver_table in-place for the idle drivers
                    simulator.driver_table.loc[is_idle, 'action_index'] = action_indices
                    simulator.driver_table.loc[is_idle, 'matching_radius'] = matching_radii

                    # observe the transition and store the transition in the replay buffer (simulator.dispatch_transitions_buffer)
                    simulator.step(agent)

                    if len(simulator.replay_buffer) >= BATCH_SIZE and step % 100 == 0:
                        states, action_indices, rewards, next_states= simulator.replay_buffer.sample(BATCH_SIZE)
                        agent.learn(states, action_indices, rewards, next_states)
                end_time = time.time()
                total_reward_record[epoch] = simulator.total_reward

                print('epoch:', epoch)
                print('epoch running time: ', end_time - start_time)
                print('epoch total reward: ', simulator.total_reward)
                print("total orders", simulator.total_request_num)
                print("matched orders", simulator.matched_requests_num)
                if args.rl_agent == "a2c":
                    print("policy loss", agent.policy_loss_list)
                    print("value loss", agent.value_loss_list)

                training_log['epoch_running_times'].append(end_time - start_time)
                training_log['epoch_total_rewards'].append(simulator.total_reward)
                training_log['epoch_total_orders'].append(simulator.total_request_num)
                training_log['epoch_matched_orders'].append(simulator.matched_requests_num)

                if epoch % 5 == 0:
                    if args.rl_agent == "dqn":
                        with open(
                                f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}.pickle",
                                'wb') as f:
                            pickle.dump(training_log, f)
                        with open(
                                f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_losses.pickle",
                                "wb") as f:
                            pickle.dump(agent.loss_values, f)
                        agent.save_parameters(parameter_path)
                    else:
                        with open(
                                f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}.pickle",
                                'wb') as f:
                            pickle.dump(training_log, f)
                        with open(
                                f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_policy_loss.pickle",
                                "wb") as f:
                            pickle.dump(agent.policy_loss_list, f)
                        with open(
                                f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_value_loss.pickle",
                                "wb") as f:
                            pickle.dump(agent.value_loss_list, f)
                        # agent.save_parameters("pre_trained/a2c/")

                if args.rl_agent == "dqn":
                    with open(
                            f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}.pickle",
                            'wb') as f:
                        pickle.dump(training_log, f)
                    with open(
                            f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_losses.pickle",
                            "wb") as f:
                        pickle.dump(agent.loss_values, f)
                    agent.save_parameters(parameter_path)
                else:
                    with open(
                            f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}.pickle",
                            'wb') as f:
                        pickle.dump(training_log, f)
                    with open(
                            f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_policy_loss.pickle",
                            "wb") as f:
                        pickle.dump(agent.policy_loss_list, f)
                    with open(
                            f"./training_log/{args.rl_agent}_{'_'.join(map(str, args.layers_dimension_list))}_actionspace_{len(args.action_space)}_value_loss.pickle",
                            "wb") as f:
                        pickle.dump(agent.value_loss_list, f)
