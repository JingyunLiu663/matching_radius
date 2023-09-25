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

def worker_process(args):
    agent, simulator, state = args
    action_index = agent.choose_action(state)
    return action_index, env_params['radius_action_space'][action_index]

if __name__ == "__main__":
    driver_num = 100
    max_distance_num = 1
    cruise_flag = True  # taking cruise into consideration
    pickup_flag = 'rg'
    delivery_flag = 'rg'
    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    track_record = []

    # initialize the simulator
    simulator = Simulator(**env_params)

    if env_params['rl_mode'] == "matching_radius":
        if simulator.experiment_mode == 'train':
            print("training process:")
            # initialize the RL agent for matching radius setting
            agent = DqnAgent(**dqn_params)
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
                    idle_driver_table = simulator.driver_table[
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
                        simulator.driver_table['matching_radius'] = env_params['radius_action_space'][action_index]
                    # observe the transition and store the transition in the replay buffer
                  
                    transition_buffer = simulator.step()
                    # print("simulator step finished")
                    if transition_buffer:
                        states, action_indices, rewards, next_states = transition_buffer
                        agent.store_transition(states, action_indices, rewards, next_states)
                    # print("store_transition finished")
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
                # print("step1:order dispatching:", simulator.time_step1)
                # print("step2:reaction", simulator.time_step2)
                # print("step3:bootstrap new orders:", simulator.step3)
                # print("step4:cruise:", simulator.step4)
                # print("step4_1:track_recording", simulator.step4_1)
                # print("step5:update state", simulator.step5)
                # print("step6:offline update", simulator.step6)
                # print("step7: update time", simulator.step7)

                with open("output3/order_record-1103.pickle", "wb") as f:
                    pickle.dump(simulator.record, f)
                # if epoch % 200 == 0:  # save the result every 200 epochs
                #     agent.save_parameters(epoch)

                if epoch % 200 == 0:  # plot and save training curve
                    # plt.plot(list(range(epoch)), total_reward_record[:epoch])
                    with open(load_path + 'training_results_record', 'wb') as f:
                        pickle.dump(total_reward_record, f)
