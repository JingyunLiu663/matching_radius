from config import *
import numpy as np
import time
from training import *

def simulation_test(args, agent, simulator, logger, test_num=10):

    for num in range(test_num):
        print('test round: ', num)

        for date in TEST_DATE_LIST:
            simulator.experiment_date = date
            simulator.reset()

            for _ in range(simulator.finish_run_step):
                if simulator.wait_requests.shape[0] > 0:
                    states = get_states(simulator)
                    action_indices, matching_radius = get_actions_given_states(agent, states, args)
                    simulator.wait_requests['action_index'] = action_indices
                    simulator.wait_requests['matching_radius'] = matching_radius
                simulator.step()  # observe the transition

        logger.info(f"epoch : {num} == total_reward:{simulator.total_reward},matching_rate:{simulator.matched_requests_num/simulator.total_request_num},total_request_num:{simulator.total_request_num},matched_request_num:{simulator.matched_requests_num},occupancy_rate:{simulator.occupancy_rate},occupancy_rate_no_pickup:{simulator.occupancy_rate_no_pickup},wait_time:{simulator.waiting_time / simulator.matched_requests_num},pickup_time:{simulator.pickup_time / simulator.matched_requests_num}")
