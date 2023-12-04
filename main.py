from simulator_env import Simulator
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
import torch

warnings.filterwarnings("ignore")
import os
from utilities import *
from dqn import DqnAgent
from ddqn import DDqnAgent
from dueling_dqn import DuelingDqnAgent
import config
from matplotlib import pyplot as plt
import argparse
from train_test_log_generation import TrainTestLog, EpochLog
from draw_snapshots import draw_simulation



def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-radius', type=float, default=3.5, help="max pickup radius")
    parser.add_argument('-rl_agent', type=str, default="dqn", help='RL agent') 
    parser.add_argument('-action_space', type=float, nargs='+',
                        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
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
    parser.add_argument('-rl_mode', type=str, default="rl", help='rl/rl_greedy/fixed') #0: False 1: True
    parser.add_argument('-experiment_mode', type=str, default="train", help="train/test")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU is available")
    # get command line arguments
    args = get_args()
    print("action space:", args.action_space) 

    # initialize the simulator
    simulator = Simulator(**env_params)
    simulator.adjust_reward_by_radius = args.adjust_reward
    simulator.experiment_mode = args.experiment_mode
    simulator.rl_mode = args.rl_mode
    simulator.maximal_pickup_distance = args.radius # for rl_mode == fixed

    if simulator.adjust_reward_by_radius: 
        print("reward adjusted by radius")
    else:
        print("reward not adjusted by radius")

    if simulator.experiment_mode == 'train' and simulator.rl_mode == "rl":
        print("RL training process:")
        train_log = TrainTestLog()

        # initialize the RL agent for matching radius 
        if args.rl_agent == "dqn":
            agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("dqn agent is created")
        elif args.rl_agent == "ddqn":
            agent = DDqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("double dqn agent is created")
        elif args.rl_agent == "dueling_dqn":
            agent = DuelingDqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("dueling dqn agent is created")

        # save and load pre-trained model
        parameter_path = (f"pre_trained/{args.rl_agent}/{args.rl_agent}_epoch119_{args.adjust_reward}_model.pth")
        if env_params['pre_trained']:
            agent.load_parameters(parameter_path)
            print("pre-trained model is loaded")

        epoch_log = EpochLog(simulator.experiment_mode)

        for epoch in range(NUM_EPOCH):
            # clear former epoch log
            epoch_log.reset()
            # initialize replay_buffer
            replay_buffer = ReplayBuffer()

            for date in TRAIN_DATE_LIST:
                # initialize the environment - simulator.driver_table is constructed (a deep copy of sampled simulator.driver_info)
                simulator.experiment_date = date
                simulator.reset()
                
                start_time = time.time()
                # for every time interval do:
                for step in range(simulator.finish_run_step):
                    if simulator.wait_requests.shape[0] > 0:
                        simulator.get_demand_supply()
                        # reindex by order_id
                        idle_drivers = simulator.idle_drivers_per_grid.reindex(simulator.wait_requests['order_id']).fillna(0)
                        open_orders = simulator.open_orders_per_grid.reindex(simulator.wait_requests['order_id']).fillna(0)

                        states = list(zip(simulator.wait_requests['order_id'], 
                                        [simulator.time]*len(simulator.wait_requests), 
                                        idle_drivers, 
                                        open_orders, 
                                        simulator.wait_requests['wait_time']))

                        action_indices = [agent.choose_action(state) for state in states]
                        simulator.wait_requests['action_index'] = action_indices
                        simulator.wait_requests['matching_radius'] = [args.action_space[i] for i in action_indices]
                    
                        # store state and action in ongoing trajectories
                        for order_id, state, action_index in zip(simulator.wait_requests['order_id'], states, action_indices):
                            if order_id not in simulator.orders.trajectories:
                                simulator.orders.trajectories[order_id] = []
                            simulator.orders.trajectories[order_id].append((state, action_index))

                    # observe the transition and store the transition in the replay buffer 
                    simulator.step()

                    # if any orders have reached a terminal state, add their trajectories to the replay buffer
                    for order_id, rewards in simulator.terminal_orders_with_rewards.items():
                        trajectory = simulator.orders.trajectories.pop(order_id)
                        for i, ((state, action), reward) in enumerate(zip(trajectory, rewards)):
                            if i < len(trajectory) - 1:
                                next_state = trajectory[i+1][0]
                                done = False
                            else:
                                next_state = np.zeros(5)
                                done = True
                            replay_buffer.push((state, action, reward, next_state, done))
                    simulator.terminal_orders_with_rewards = {}

                    if len(replay_buffer) >= BATCH_SIZE:
                        for i in range(10):
                            states, action_indices, rewards, next_states, done = replay_buffer.sample(BATCH_SIZE)
                            agent.learn(states, action_indices, rewards, next_states, done)
                            # keep track of the loss per time step
                            epoch_log.add_loss_per_learning_step(agent.loss)
                end_time = time.time()

                print(f'epoch: {epoch} date:{date}')
                print('epoch running time: ', end_time - start_time)
                print('epoch average loss: ', np.mean(epoch_log.epoch_loss))
                print('epoch total reward: ', simulator.total_reward)
                print('total orders', simulator.total_request_num)
                print('matched orders', simulator.matched_requests_num)
                print('total adjusted reward', simulator.total_reward_per_pickup_dist)
                print('matched order ratio', simulator.matched_requests_num / simulator.total_request_num)

                epoch_log.add_within_epoch(simulator.total_reward, simulator.total_reward_per_pickup_dist, 
                                           simulator.total_request_num, simulator.matched_requests_num,
                                           simulator.occupancy_rate, simulator.occupancy_rate_no_pickup,
                                           simulator.pickup_time, simulator.waiting_time, end_time - start_time)
            
            # add scalar to TensorBoard
            epoch_log.add_to_tensorboard(agent, epoch)

            # store in record dict  
            epoch_log.add_to_pickle_file(train_log.record_dict)

            # save RL model parameters
            if not os.path.exists(f"pre_trained/{args.rl_mode}"):
                os.makedirs(f"pre_trained/{args.rl_mode}")
            if (epoch > 95 and epoch % 10 == 0) or epoch == NUM_EPOCH - 1:
                parameter_path = (f"pre_trained/{args.rl_mode}/{args.rl_agent}_epoch{epoch}_{args.adjust_reward}_model.pth")
                agent.save_parameters(parameter_path)
                
        # serialize the record
        train_log.save(adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode,
                       rl_agent=args.rl_agent,  actions=simulator.action_collection)
                    
        # close TensorBoard writer
        agent.train_writer.close()


    elif simulator.rl_mode == "rl" and simulator.experiment_mode == 'test':
        print("RL - testing process:")
        test_log = TrainTestLog()

        if args.rl_agent == "dqn":
            agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("dqn agent is created")
        elif args.rl_agent == "ddqn":
            agent = DDqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("double dqn agent is created")
        elif args.rl_agent == "dueling_dqn":
            agent = DuelingDqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("dueling dqn agent is created")

        parameter_path = (f"pre_trained/{args.rl_mode}/"
                            f"{args.rl_agent}_epoch119_{args.adjust_reward}_model.pth")
        agent.load_parameters(parameter_path)
        print("RL agent parameter loaded")

        epoch_log = EpochLog(simulator.experiment_mode)
        test_num = 10
        for num in range(test_num):
            print('test round: ', num)
            # clear former epoch log
            epoch_log.reset()

            for date in TEST_DATE_LIST:
                simulator.experiment_date = date
                simulator.reset()

                start_time = time.time()
                for step in range(simulator.finish_run_step):
                    # Fetch grid_ids for the idle drivers
                    grid_ids = simulator.wait_requests['origin_grid_id'].values.reshape(-1, 1)
                    time_slices = np.full_like(grid_ids, simulator.time).reshape(-1, 1)
                    # Determine the action_indices for the idle drivers
                    states_array = np.hstack((time_slices, grid_ids)).astype(np.float32)
                    action_indices = agent.choose_action(states_array)

                    # Calculate matching radius for the idle drivers
                    action_space_array = np.array(args.action_space)
                    matching_radii = action_space_array[action_indices]

                    # Update the simulator.driver_table in-place for the idle drivers
                    simulator.wait_requests['action_index'] = action_indices
                    simulator.wait_requests['matching_radius'] = matching_radii

                    # observe the transition and store the transition in the replay buffer (simulator.dispatch_transitions_buffer)
                    simulator.step()
                end_time = time.time()

                epoch_log.add_within_epoch(simulator.total_reward, simulator.total_reward_per_pickup_dist, 
                                            simulator.total_request_num, simulator.matched_requests_num,
                                            simulator.occupancy_rate, simulator.occupancy_rate_no_pickup,
                                            simulator.pickup_time, simulator.waiting_time, end_time - start_time)
            
            print("total reward", sum(epoch_log.total_reward))
            print("total adjusted reeward", sum(epoch_log.total_adjusted_reward))
            print("pick", np.mean(epoch_log.pickup_time))
            print("wait", np.mean(epoch_log.waiting_time))
            print("matching ratio", sum(epoch_log.matched_requests_num)/ sum(epoch_log.total_request_num))
            print("ocu rate", np.mean(epoch_log.occupancy_rate))

            # add scalar to TensorBoard
            epoch_log.add_to_tensorboard(agent, num)
            # store in record dict  
            epoch_log.add_to_pickle_file(test_log.record_dict)

        # close TensorBoard writer
        agent.test_writer.close()

        # serialize the record
        test_log.save(adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode,
                       rl_agent=args.rl_agent,  actions=simulator.action_collection)

    elif simulator.rl_mode == "fixed":
        print("fixed radius = ", simulator.maximal_pickup_distance)
        
        test_num = 10
        date_list = None
        if simulator.experiment_mode == 'test':
            date_list = TEST_DATE_LIST
        elif simulator.experiment_mode == 'train':
            date_list = TRAIN_DATE_LIST
            
        for date in date_list:
            test_log = TrainTestLog()
            epoch_log = EpochLog(simulator.experiment_mode)
            simulator.experiment_date = date
            for num in range(test_num):
                print('test round: ', num)
                # clear former epoch log
                epoch_log.reset()
                simulator.reset()
                start_time = time.time()
                for step in range(simulator.finish_run_step):
                    # observe the transition and store the transition in the replay buffer (simulator.dispatch_transitions_buffer)
                    simulator.step()
                    if step % 300 == 0:
                        draw_simulation(simulator, "./input/graph.graphml", time_step=step)
                end_time = time.time()

                epoch_log.add_within_epoch(simulator.total_reward, simulator.total_reward_per_pickup_dist, 
                                            simulator.total_request_num, simulator.matched_requests_num,
                                            simulator.occupancy_rate, simulator.occupancy_rate_no_pickup,
                                            simulator.pickup_time, simulator.waiting_time, end_time - start_time)
                
                print("total reward", sum(epoch_log.total_reward))
                print("total adjusted reeward", sum(epoch_log.total_adjusted_reward))
                print("pick", np.mean(epoch_log.pickup_time))
                print("wait", np.mean(epoch_log.waiting_time))
                print("matching ratio", sum(epoch_log.matched_requests_num)/ sum(epoch_log.total_request_num))
                print("ocu rate", np.mean(epoch_log.occupancy_rate))

                # store in record dict  
                epoch_log.add_to_pickle_file(test_log.record_dict)

            # serialize the record
            test_log.save(adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, 
                        rl_mode=simulator.rl_mode, radius=simulator.maximal_pickup_distance, date=date)

    elif simulator.rl_mode == "greedy":
        print("greedy radius process:")
        test_log = TrainTestLog()
        epoch_log = EpochLog(args.experiment_mode)
        test_num = 10
        for num in range(test_num):
            print('test round: ', num)

            # clear former epoch log
            epoch_log.reset()

            date_list = None
            if args.experiment_mode == "train":
                date_list = TRAIN_DATE_LIST
            else:
                date_list = TEST_DATE_LIST
            
            for date in date_list:
                simulator.experiment_date = date
                simulator.reset()
                start_time = time.time()
                simulator.wait_requests['matching_radius'] = 0.5
                for step in range(simulator.finish_run_step):
                    simulator.step()
                end_time = time.time()

                epoch_log.add_within_epoch(simulator.total_reward, simulator.total_reward_per_pickup_dist, 
                                            simulator.total_request_num, simulator.matched_requests_num,
                                            simulator.occupancy_rate, simulator.occupancy_rate_no_pickup,
                                            simulator.pickup_time, simulator.waiting_time, end_time - start_time)
            
            print("total reward", sum(epoch_log.total_reward))
            print("total adjusted reeward", sum(epoch_log.total_adjusted_reward))
            print("pick", np.mean(epoch_log.pickup_time))
            print("wait", np.mean(epoch_log.waiting_time))
            print("matching ratio", sum(epoch_log.matched_requests_num)/ sum(epoch_log.total_request_num))
            print("ocu rate", np.mean(epoch_log.occupancy_rate))

            # store in record dict  
            epoch_log.add_to_pickle_file(test_log.record_dict)

    # serialize the record
    test_log.save(rl_agent=None, adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode, actions=simulator.action_collection)