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
from ddqn import DDqnAgent
from a2c import A2CAgent
import config
from matplotlib import pyplot as plt
import argparse
import optuna
from optuna.integration import TensorBoardCallback


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-rl_agent', type=str, default="dqn", help='RL agent') # "dqn" "a2c"
    parser.add_argument('-action_space', type=float, nargs='+',
                        default=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
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

    # 0: False 1:True
    parser.add_argument('-adjust_reward', type=int, default=0, 
                        help='apply immediate reward(0), or adjust the reward by matching radius(1)') 
    parser.add_argument('-rl_mode', type=str, default="matching_radius", help='maching_radius/random') #0: False 1: True
    parser.add_argument('-experiment_mode', type=str, default="train", help="train/test")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # get command line arguments
    args = get_args()
    print("action space:", args.action_space) 

    # initialize the simulator
    simulator = Simulator(**env_params)
    simulator.adjust_reward_by_radius = args.adjust_reward
    simulator.experiment_mode = args.experiment_mode
    simulator.rl_mode = args.rl_mode

    if simulator.experiment_mode == 'train' and simulator.rl_mode == "matching_radius":
        print("training process:")
        # log record for evaluation metrics
        column_list = ['epoch_average_loss', 'total_adjusted_reward', 'total_reward',
               'total_request_num', 'matched_request_num',
               'matched_request_ratio',
               'waiting_time', 'pickup_time',
               'occupancy_rate','occupancy_rate_no_pickup']
        
        record_dict = {col: [] for col in column_list}
        # initialize the RL agent for matching radius 
        if args.rl_agent == "dqn":
            agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, 2000, args.experiment_mode, args.adjust_reward)
            print("dqn agent is created")
        elif args.rl_agent == "ddqn":
            agent = DDqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec, 2000, args.experiment_mode, args.adjust_reward)
            print("double dqn agent is created")

        #  # use pre-trained model
        # if env_params['pre_trained']:
        #     agent.load_parameters(parameter_path)
        #     print("pre-trained model is loaded")

        for epoch in range(NUM_EPOCH):
            # each epoch walks through 5 weekdays in a row
            episode_time = []
            epoch_loss = []
            episode_reward = []
            episode_adjusted_reward = []
            episode_total_request_num = []
            episode_matched_requests_num = []
            episode_occupancy_rate = []
            episode_occupancy_rate_no_pickup = []
            episode_pickup_time = []
            episode_waiting_time = []

            for date in TRAIN_DATE_LIST:
                # initialize the environment - simulator.driver_table is constructed (a deep copy of sampled simulator.driver_info)
                simulator.experiment_date = date
                simulator.reset()
                if simulator.adjust_reward_by_radius: 
                    print("reward adjusted by radius")
                else:
                    print("reward not adjusted by radius")

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

                    # log the action indices distribution
                    # agent.train_writer.add_histogram('action_indices_distribution', action_indices, agent.eval_net_update_times)

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
                        print(f"epoch: {epoch} date: {date} step: {step} loss: {agent.loss}")
                        epoch_loss.append(agent.loss)
                end_time = time.time()

                print(f'epoch: {epoch} date:{date}')
                print('epoch running time: ', end_time - start_time)
                print('epoch average loss: ', np.mean(epoch_loss))
                print('epoch total reward: ', simulator.total_reward)
                print('total orders', simulator.total_request_num)
                print('matched orders', simulator.matched_requests_num)
                print('total adjusted reward', simulator.total_reward_per_pickup_dist)
                print('matched order ratio', simulator.matched_requests_num / simulator.total_request_num)

                episode_time.append(end_time - start_time)
                episode_adjusted_reward.append(simulator.total_reward_per_pickup_dist)
                episode_reward.append(simulator.total_reward)
                episode_total_request_num.append(simulator.total_request_num)
                episode_matched_requests_num.append(simulator.matched_requests_num)
                episode_occupancy_rate.append(simulator.occupancy_rate)
                episode_occupancy_rate_no_pickup.append(simulator.occupancy_rate_no_pickup)
                episode_pickup_time.append(simulator.pickup_time)
                episode_waiting_time.append(simulator.waiting_time)
            
            # add scalar to TensorBoard
            '''
            agent.train_writer.add_scalar('epoch running time', np.mean(episode_time), epoch)
            agent.train_writer.add_scalar('epoch average loss', np.mean(epoch_loss), epoch)
            agent.train_writer.add_scalar('epoch total adjusted reward (per pickup distance)', np.mean(episode_adjusted_reward), epoch)
            agent.train_writer.add_scalar('epoch total reward', np.mean(episode_reward), epoch)
            agent.train_writer.add_scalar('total orders', np.mean(episode_total_request_num), epoch)
            agent.train_writer.add_scalar('matched orders', np.mean(episode_matched_requests_num), epoch)
            agent.train_writer.add_scalar('matched request ratio', sum(episode_matched_requests_num)/ sum(episode_total_request_num), epoch)
            agent.train_writer.add_scalar('matched occupancy rate', np.mean(episode_occupancy_rate), epoch)
            agent.train_writer.add_scalar('matched occupancy rate - no pickup', np.mean(episode_occupancy_rate_no_pickup), epoch)
            agent.train_writer.add_scalar('pickup time', np.mean(episode_pickup_time), epoch)
            agent.train_writer.add_scalar('waiting time', np.mean(episode_waiting_time), epoch)
            '''

            # store in record dict  
            print(f"epoch: {epoch}")    
            print(len(epoch_loss))   
            print("1st element", epoch_loss[0])   
            print("last element", epoch_loss[-1])   
            record_dict['epoch_average_loss'].append(np.mean(epoch_loss))
            record_dict['total_adjusted_reward'].append(np.mean(episode_adjusted_reward))
            record_dict['total_reward'].append(np.mean(episode_reward))
            record_dict['total_request_num'].append(np.mean(episode_total_request_num))
            record_dict['matched_request_num'].append(np.mean(episode_matched_requests_num))
            record_dict['matched_request_ratio'].append(sum(episode_matched_requests_num) / sum(episode_total_request_num))
            record_dict['occupancy_rate'].append(np.mean(episode_occupancy_rate))
            record_dict['occupancy_rate_no_pickup'].append(np.mean(episode_occupancy_rate_no_pickup))
            record_dict['pickup_time'].append(np.mean(episode_pickup_time))
            record_dict['waiting_time'].append(np.mean(episode_waiting_time))

            if epoch % 10 == 0:
                # save RL model parameters
                parameter_path = (f"pre_trained/{args.rl_agent}/{args.rl_agent}_epoch{epoch}_{args.adjust_reward}_model.pth")
                agent.save_parameters(parameter_path)

                 # serialize the record
                if simulator.adjust_reward_by_radius:
                    with open(f'training_record_{args.rl_agent}_adjusted_reward.pkl', 'wb') as f:
                        pickle.dump(record_dict, f)
                else:
                    with open(f'training_record_{args.rl_agent}_immediate_reward.pkl', 'wb') as f:
                        pickle.dump(record_dict, f)
        # close TensorBoard writer
        agent.train_writer.close()
       

    elif simulator.experiment_mode == 'test' and simulator.rl_mode == "matching_radius":
        print("testing process:")
        column_list = ['total_adjusted_reward', 'total_reward',
               'total_request_num', 'matched_request_num',
               'matched_request_ratio',
               'waiting_time', 'pickup_time',
               'occupancy_rate','occupancy_rate_no_pickup']
        
        record_dict = {col: [] for col in column_list}

        test_num = 10
      
        for num in range(test_num):
            print('num: ', num)
            if args.rl_agent == "dqn":
                agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, args.lr, args.gamma,
                                args.epsilon, args.eps_min, args.eps_dec)
                parameter_path = (f"pre_trained/{args.rl_agent}/"
                                f"{args.rl_agent}_{'_'.join(map(str, args.dim_list))}_model.pth")
                print("dqn agent is created")
            agent.load_parameters(parameter_path)
            print("RL agent parameter loaded")

            total_adjusted_reward = []
            total_reward = []
            total_request_num = []
            matched_request_num = []
            occupancy_rate = []
            occupancy_rate_no_pickup = []
            pickup_time = []
            waiting_time = []

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

                    # Log the action indices distribution
                    # agent.writer.add_histogram('action_indices_distribution', action_indices, agent.eval_net_update_times)

                    # Calculate matching radius for the idle drivers
                    action_space_array = np.array(args.action_space)
                    matching_radii = action_space_array[action_indices]

                    # Update the simulator.driver_table in-place for the idle drivers
                    simulator.driver_table.loc[is_idle, 'action_index'] = action_indices
                    simulator.driver_table.loc[is_idle, 'matching_radius'] = matching_radii

                    # observe the transition and store the transition in the replay buffer (simulator.dispatch_transitions_buffer)
                    simulator.step()
                end_time = time.time()

                total_reward.append(simulator.total_reward)
                total_adjusted_reward.append(simulator.total_reward_per_pickup_dist)
                total_request_num.append(simulator.total_request_num)
                occupancy_rate.append(simulator.occupancy_rate)
                matched_request_num.append(simulator.matched_requests_num)
                occupancy_rate_no_pickup.append(simulator.occupancy_rate_no_pickup)
                pickup_time.append(simulator.pickup_time / simulator.matched_requests_num)
                waiting_time.append(simulator.waiting_time / simulator.matched_requests_num)
            
            print("total reward",np.mean(total_reward))
            print("total adjusted reeward", np.mean(total_adjusted_reward))
            print("pick",np.mean(pickup_time))
            print("wait", np.mean(waiting_time))
            print("matching ratio", sum(matched_request_num)/ sum(total_request_num))
            print("ocu rate", occupancy_rate)
            
            # add scalar to TensorBoard
            
            agent.test_writer.add_scalar('total reward', np.mean(total_reward), num)
            agent.test_writer.add_scalar('total adjusted reward', np.mean(total_adjusted_reward), num)
            agent.test_writer.add_scalar('total_request_num', np.mean(total_request_num), num)
            agent.test_writer.add_scalar('matched_request_num', np.mean(matched_request_num), num)
            agent.test_writer.add_scalar('matched_request_ratio', sum(matched_request_num)/ sum(total_request_num), num)
            agent.test_writer.add_scalar('occupancy_rate', np.mean(occupancy_rate), num)
            agent.test_writer.add_scalar('occupancy_rate_no_pickup', np.mean(occupancy_rate_no_pickup), num)
            agent.test_writer.add_scalar('pickup_time', np.mean(pickup_time), num)
            agent.test_writer.add_scalar('waiting_time', np.mean(waiting_time), num)
            
            # Add scalar to TensorBoard and store in record_dict
            record_dict['total_reward'].append(np.mean(total_reward))
            record_dict['total_adjusted_reward'].append(np.mean(total_adjusted_reward))
            record_dict['total_request_num'].append(np.mean(total_request_num))
            record_dict['matched_request_num'].append(np.mean(matched_request_num))
            record_dict['matched_request_ratio'].append(sum(matched_request_num) / sum(total_request_num))
            record_dict['occupancy_rate'].append(np.mean(occupancy_rate))
            record_dict['occupancy_rate_no_pickup'].append(np.mean(occupancy_rate_no_pickup))
            record_dict['pickup_time'].append(np.mean(pickup_time))
            record_dict['waiting_time'].append(np.mean(waiting_time))
           
        # close TensorBoard writer
        agent.test_writer.close()
        # serialize the testing records
        if simulator.adjust_reward_by_radius:
            with open(f'testing_record_{args.rl_agents}_adjusted_reward.pkl', 'wb') as f:
                pickle.dump(record_dict, f)
        else:
            with open(f'testing_record_{args.rl_agents}_immediate_reward.pkl', 'wb') as f:
                pickle.dump(record_dict, f)

    elif simulator.rl_mode == "random":
        print("random process:")
        column_list = ['total_adjusted_reward', 'total_reward',
               'total_request_num', 'matched_request_num',
               'matched_request_ratio',
               'waiting_time', 'pickup_time',
               'occupancy_rate','occupancy_rate_no_pickup']
        
        record_dict = {col: [] for col in column_list}

        test_num = 10
      
        for num in range(test_num):
            print('num: ', num)

            total_adjusted_reward = []
            total_reward = []
            total_request_num = []
            matched_request_num = []
            occupancy_rate = []
            occupancy_rate_no_pickup = []
            pickup_time = []
            waiting_time = []

            for date in TEST_DATE_LIST:
                simulator.experiment_date = date
                simulator.reset()
                start_time = time.time()
                for step in range(simulator.finish_run_step):
                    # Get the boolean mask for idle drivers
                    is_idle = (simulator.driver_table['status'] == 0) | (simulator.driver_table['status'] == 4)
                    simulator.driver_table.loc[is_idle, 'matching_radius'] = np.random.choice(args.action_space, np.sum(is_idle))
                    # observe the transition and store the transition in the replay buffer (simulator.dispatch_transitions_buffer)
                    simulator.step()
                end_time = time.time()

                total_reward.append(simulator.total_reward)
                total_adjusted_reward.append(simulator.total_reward_per_pickup_dist)
                total_request_num.append(simulator.total_request_num)
                occupancy_rate.append(simulator.occupancy_rate)
                matched_request_num.append(simulator.matched_requests_num)
                occupancy_rate_no_pickup.append(simulator.occupancy_rate_no_pickup)
                pickup_time.append(simulator.pickup_time / simulator.matched_requests_num)
                waiting_time.append(simulator.waiting_time / simulator.matched_requests_num)
            
            print("total reward",np.mean(total_reward))
            print("total adjusted reeward", np.mean(total_adjusted_reward))
            print("pick",np.mean(pickup_time))
            print("wait", np.mean(waiting_time))
            print("matching ratio", sum(matched_request_num)/ sum(total_request_num))
            print("ocu rate", occupancy_rate)
            
            # Add scalar to TensorBoard and store in record_dict
            record_dict['total_reward'].append(np.mean(total_reward))
            record_dict['total_adjusted_reward'].append(np.mean(total_adjusted_reward))
            record_dict['total_request_num'].append(np.mean(total_request_num))
            record_dict['matched_request_num'].append(np.mean(matched_request_num))
            record_dict['matched_request_ratio'].append(sum(matched_request_num) / sum(total_request_num))
            record_dict['occupancy_rate'].append(np.mean(occupancy_rate))
            record_dict['occupancy_rate_no_pickup'].append(np.mean(occupancy_rate_no_pickup))
            record_dict['pickup_time'].append(np.mean(pickup_time))
            record_dict['waiting_time'].append(np.mean(waiting_time))

            # serialize the testing records
        if simulator.adjust_reward_by_radius:
            with open(f'fixed_radius_adjusted_reward_test.pkl', 'wb') as f:
                pickle.dump(record_dict, f)
        else:
            with open(f'fixed_radius_immediate_reward_test.pkl', 'wb') as f:
                pickle.dump(record_dict, f)
    else:
        pass