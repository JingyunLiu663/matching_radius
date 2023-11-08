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



def get_args():
    parser = argparse.ArgumentParser()

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

    # 0: False 1:True
    parser.add_argument('-adjust_reward', type=int, default=0, 
                        help='apply immediate reward(0), or adjust the reward by matching radius(1)') 
    parser.add_argument('-rl_mode', type=str, default="rl_1stage", help='random') #0: False 1: True
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

    if simulator.adjust_reward_by_radius: 
        print("reward adjusted by radius")
    else:
        print("reward not adjusted by radius")

    if simulator.experiment_mode == 'train' and simulator.rl_mode == "rl_1stage":
        print("training process:")
        train_log = TrainTestLog()
        # initialize the RL agent for matching radius 
        if args.rl_agent == "dqn":
            agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("dqn agent is created")
        elif args.rl_agent == "ddqn":
            agent = DDqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("double dqn agent is created")
        elif args.rl_agent == "dueling_dqn":
            agent = DuelingDqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
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

            for date in TRAIN_DATE_LIST:
                # initialize the environment - simulator.driver_table is constructed (a deep copy of sampled simulator.driver_info)
                simulator.experiment_date = date
                simulator.reset()

                start_time = time.time()
                # for every time interval do:
                for step in range(simulator.finish_run_step):
                    # get the boolean mask for idle drivers
                    is_idle = (simulator.driver_table['status'] == 0) | (simulator.driver_table['status'] == 4)
                    # fetch grid_ids for the idle drivers
                    grid_ids = simulator.driver_table.loc[is_idle, 'grid_id'].values.reshape(-1, 1)
                    time_slices = np.full_like(grid_ids, simulator.time).reshape(-1, 1)

                    # determine the action_indices for the idle drivers
                    states_array = np.hstack((time_slices, grid_ids)).astype(np.float32)
                    action_indices = agent.choose_action(states_array)

                    #log the action indices distribution
                    agent.train_writer.add_histogram('action_indices_distribution', action_indices, agent.eval_net_update_times)

                    # calculate matching radius for the idle drivers
                    action_space_array = np.array(args.action_space)
                    matching_radii = action_space_array[action_indices]

                    # update the simulator.driver_table in-place for the idle drivers
                    simulator.driver_table.loc[is_idle, 'action_index'] = action_indices
                    simulator.driver_table.loc[is_idle, 'matching_radius'] = matching_radii

                    # observe the transition and store the transition in the replay buffer (simulator.dispatch_transitions_buffer)
                    simulator.step()

                    if len(simulator.replay_buffer) >= BATCH_SIZE:
                        states, action_indices, rewards, next_states= simulator.replay_buffer.sample(BATCH_SIZE)
                        agent.learn(states, action_indices, rewards, next_states)
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
            # if (epoch > 95 and epoch % 10 == 0) or epoch == NUM_EPOCH - 1:
            #     parameter_path = (f"pre_trained/{args.rl_agent}/{args.rl_agent}_epoch{epoch}_{args.adjust_reward}_model.pth")
            #     agent.save_parameters(parameter_path)
                
        # serialize the record
        # train_log.save(args.rl_agent, adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode)
                    
        # close TensorBoard writer
        agent.train_writer.close()


    elif simulator.experiment_mode == 'test' and simulator.rl_mode == "rl_1stage":
        print("testing process:")
        test_log = TrainTestLog()

        if args.rl_agent == "dqn":
            agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("dqn agent is created")
        elif args.rl_agent == "ddqn":
            agent = DDqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("double dqn agent is created")
        elif args.rl_agent == "dueling_dqn":
            agent = DuelingDqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
            print("dueling dqn agent is created")

        parameter_path = (f"pre_trained/{args.rl_agent}/"
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
                    # Get the boolean mask for idle drivers
                    is_idle = (simulator.driver_table['status'] == 0) | (simulator.driver_table['status'] == 4)
                    # Fetch grid_ids for the idle drivers
                    grid_ids = simulator.driver_table.loc[is_idle, 'grid_id'].values.reshape(-1, 1)
                    time_slices = np.full_like(grid_ids, simulator.time).reshape(-1, 1)
                    # Determine the action_indices for the idle drivers
                    states_array = np.hstack((time_slices, grid_ids)).astype(np.float32)
                    action_indices = agent.choose_action(states_array)

                    # Calculate matching radius for the idle drivers
                    action_space_array = np.array(args.action_space)
                    matching_radii = action_space_array[action_indices]

                    # Update the simulator.driver_table in-place for the idle drivers
                    simulator.driver_table.loc[is_idle, 'action_index'] = action_indices
                    simulator.driver_table.loc[is_idle, 'matching_radius'] = matching_radii

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
        test_log.save(args.rl_agent, adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode)

    elif simulator.experiment_mode == 'test' and simulator.rl_mode == "rl_greedy":
        print("rl model fixed + greedy radius:")
        test_log = TrainTestLog()

        if args.rl_agent == "dqn":
                agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward, simulator.rl_mode)
                print("dqn agent is created")
        elif args.rl_agent == "ddqn":
            agent = DDqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward, simulator.rl_mode)
            print("double dqn agent is created")
        elif args.rl_agent == "dueling_dqn":
            agent = DuelingDqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward, simulator.rl_mode)
            print("dueling dqn agent is created")

        # parameter_path = (f"pre_trained/{args.rl_agent}/"
        #                     f"{args.rl_agent}_epoch119_{args.adjust_reward}_model.pth")
        parameter_path = (f"pre_trained/temp/{args.rl_agent}_epoch119_{args.adjust_reward}_model.pth")
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
                simulator.driver_table['matching_radius'] = 0.5

                start_time = time.time()
                for step in range(simulator.finish_run_step):
                    # 1. for those agents who has reached the maximum radius accumulate time -> set the total_idle_time to be 0
                    simulator.driver_table.loc[simulator.driver_table['matching_radius'] > 6, 'total_idle_time'] = 0

                    # 2. distinguish between radius assignment with rl model and greedy strategy
                    is_idle = (simulator.driver_table['status'] == 0) | (simulator.driver_table['status'] == 4)
                    apply_rl = is_idle & simulator.driver_table['total_idle_time'] == 0  # just become idle (parking, cruising or repositioning)
                    apply_greedy = is_idle & simulator.driver_table['total_idle_time'] > 0 # has been idle for a while (not able to match with an order in previous dispatching)

                    # 2.1. apply the RL model for radius assignment
                    # Fetch grid_ids for the idle drivers
                    grid_ids = simulator.driver_table.loc[apply_rl, 'grid_id'].values.reshape(-1, 1)
                    time_slices = np.full_like(grid_ids, simulator.time).reshape(-1, 1)
                    # Determine the action_indices for the idle drivers
                    states_array = np.hstack((time_slices, grid_ids)).astype(np.float32)
                    action_indices = agent.choose_action(states_array)
                    # Calculate matching radius for the idle drivers
                    action_space_array = np.array(args.action_space)
                    matching_radii = action_space_array[action_indices]

                    # Update the simulator.driver_table in-place for the idle drivers
                    simulator.driver_table.loc[apply_rl, 'action_index'] = action_indices
                    simulator.driver_table.loc[apply_rl, 'matching_radius'] = matching_radii
                    
                    # 2.2 apply greedy incremental radius strategy for radius assignment
                    #   upper boundary controlled by simulator.maximum_radius_accumulate_time_interval
                    simulator.driver_table.loc[apply_greedy, 'matching_radius'] += 0.5
                    
                    # 3. run one step of simulator
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
        test_log.save(args.rl_agent, adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode)
        
        with open(f"rl_greedy_action_collection_{simulator.rl_mode}_{args.rl_agent}_{simulator.adjust_reward_by_radius}_{simulator.maximum_radius_accumulate_time_interval}.pkl", "wb") as f:
            pickle.dump(simulator.action_collection, f)
    
    elif simulator.rl_mode == "random":
        print("random process:")
        test_log = TrainTestLog()

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
                    # Get the boolean mask for idle drivers
                    is_idle = (simulator.driver_table['status'] == 0) | (simulator.driver_table['status'] == 4)
                    actions = np.random.choice(args.action_space, np.sum(is_idle))
                    simulator.driver_table.loc[is_idle, 'matching_radius'] = actions
                    
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

            # store in record dict  
            epoch_log.add_to_pickle_file(test_log.record_dict)

        # serialize the record
        test_log.save(rl_agent=None, adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode)

    elif simulator.rl_mode == "greedy":
        print("greedy radius process:")
        test_log = TrainTestLog()
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
                simulator.driver_table['matching_radius'] = 0.5
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
        test_log.save(rl_agent=None, adjust_reward=simulator.adjust_reward_by_radius, experiment_mode=simulator.experiment_mode, rl_mode=simulator.rl_mode)