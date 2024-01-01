from log_generation import EpochPerformanceTracker, ActionTracker
from config import *
import numpy as np
import time
from training import *

def simulation_test(args, agent, simulator, test_num):
    performance_tracker = EpochPerformanceTracker(simulator.experiment_mode, simulator.radius_method)
    actions_dist = ActionTracker()

    for num in range(test_num):
        print('test round: ', num)
        performance_tracker.reset()
        actions_dist.init_new_epoch(num)

        for date in TEST_DATE_LIST:
            simulator.experiment_date = date
            simulator.reset()
            actions_dist.init_new_date(num, date)

            start_time = time.time()
            for _ in range(simulator.finish_run_step):
                if simulator.wait_requests.shape[0] > 0:
                    states = get_states(simulator)
                    action_indices, matching_radius = get_actions_given_states(agent, states, args)
                    actions_dist.insert_time_step(args, action_indices)
                    simulator.wait_requests['action_index'] = action_indices
                    simulator.wait_requests['matching_radius'] = matching_radius
                simulator.step()  # observe the transition
            actions_dist.end_time_step(num, date)
            end_time = time.time()

        performance_tracker.add_within_epoch(simulator.total_reward, simulator.total_reward_per_pickup_dist, 
                                    simulator.total_request_num, simulator.matched_requests_num,
                                    simulator.occupancy_rate, simulator.occupancy_rate_no_pickup,
                                    simulator.pickup_time, simulator.waiting_time, end_time - start_time)

        results_output(simulator, performance_tracker, start_time, end_time, num, date)
        # add scalar to TensorBoard
        performance_tracker.add_to_tensorboard(num, agent)
        # store in record dict  
        performance_tracker.add_to_metrics_data()
        # save the action distribution
        actions_dist.save_actions_distribution(args)
    # serialize the record
    performance_tracker.save(experiment_mode=simulator.experiment_mode, radius_method=simulator.radius_method, 
                    adjust_reward=simulator.adjust_reward_by_radius,  rl_agent=args.rl_agent)        