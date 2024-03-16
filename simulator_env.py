import pandas as pd
from simulator_pattern import *
from utilities import *
import warnings
import numpy as np
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
import sys
import cProfile
import pstats
from config import *
import io

class Simulator:
    def __init__(self, **kwargs):
        # basic parameters: time & sample
        self.t_initial = kwargs['t_initial']
        self.t_end = kwargs['t_end']
        self.delta_t = kwargs['delta_t']
        self.vehicle_speed = kwargs['vehicle_speed']
        self.repo_speed = kwargs.pop('repo_speed', 3)
        self.time = None
        self.current_step = None
        self.rl_mode = kwargs['rl_mode']
        self.radius_method = kwargs['radius_method']
        self.requests  = None
        self.record =""

        # order generation
        self.order_sample_ratio = kwargs['order_sample_ratio']
        self.request_interval = kwargs['request_interval']

        # wait cancel
        self.maximum_wait_time_mean = kwargs.pop('maximum_wait_time_mean', 120)
        self.maximum_wait_time_std = kwargs.pop('maximum_wait_time_std', 0)

        # driver cancel after matching based on maximal pickup distance
        self.maximal_pickup_distance = kwargs['maximal_pickup_distance']

        # track recording
        self.track_recording_flag = kwargs['track_recording_flag']
        self.new_tracks = {}
        self.match_and_cancel_track = {}
        self.passenger_track = {}

        # pattern
        self.experiment_mode = kwargs.pop('experiment_mode', 'train')
        self.experiment_date = kwargs.pop('experiment_date', '')
        self.request_file_name = kwargs['request_file_name']
        self.driver_file_name = kwargs['driver_file_name']
        pattern_params = {'request_file_name': self.request_file_name,
                          'driver_file_name': self.driver_file_name}
        
        pattern = SimulatorPattern(**pattern_params)
        self.RN = road_network()
        self.RN.load_data()
        # dispatch method
        self.dispatch_method = kwargs['dispatch_method']
        self.method = kwargs['method']

        # cruise and reposition related parameters
        self.cruise_flag = kwargs['cruise_flag']
        self.cruise_mode = kwargs['cruise_mode']
        self.max_idle_time = kwargs['max_idle_time']

        self.reposition_flag = kwargs['reposition_flag']
        self.reposition_mode = kwargs['reposition_mode']
        self.eligible_time_for_reposition = kwargs['eligible_time_for_reposition']

        # get steps
        self.finish_run_step = int((self.t_end - self.t_initial) // self.delta_t)

        # request tables
        self.request_columns = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                                'trip_distance', 'start_time', 'origin_grid_id','dest_grid_id', 'itinerary_node_list',
                                'itinerary_segment_dis_list', 'trip_time', 'cancel_prob', 't_matched',
                                'pickup_time', 'wait_time', 't_end', 'status', 'driver_id', 'maximum_wait_time', 'designed_reward',
                                'pickup_distance']
                      
        self.wait_requests = None
        self.matched_requests = None

        # driver tables
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']
        self.driver_table = None
        self.driver_sample_ratio = kwargs['driver_sample_ratio']

        # order and driver databases
        self.driver_info = pattern.driver_info
        self.driver_info['grid_id'] = self.driver_info['grid_id'].values.astype(int)
        self.request_all = pattern.request_all
        self.request_databases = None
        self.total_reward = 0
        self.cancelled_penalty = 0
        if self.rl_mode == 'reposition':
            self.reposition_method = kwargs['reposition_method']  # rl for repositioning
        self.adjust_reward_by_radius = False
        # JL
        self.total_pickup_dist = 0
        self.total_reward_per_pickup_dist = 0
        self.idle_drivers_per_grid = None
        self.open_orders_per_grid = None
        self.terminal_orders_with_rewards = {}
        self.orders = OrderTrajectory()
        self.action_collection = []

        # JL

    def initial_base_tables(self):
        """
        This function used to initial the driver table and order table
        :return: None
        """
        self.time = deepcopy(self.t_initial)
        self.current_step = int((self.time - self.t_initial) // self.delta_t)
        
        self.grid_value = {}
        # construct driver table
        self.driver_table = sample_all_drivers(self.driver_info, self.t_initial, self.t_end, self.driver_sample_ratio)
        self.driver_table['target_grid_id'] = self.driver_table['target_grid_id'].values.astype(int)
        if self.rl_mode == 'reposition':
            # rl for repositioning
            # drivers that are repositioning
            self.state_grid_array = np.array([])
            self.state_time_array = np.array([])
            self.action_array = np.array([])
            self.next_state_grid_array = np.array([])
            self.next_state_time_array = np.array([])
            if self.reposition_method == 'A2C_global_aware' or self.reposition_method == 'A2C':
                self.global_time = []
                self.global_drivers_num = []
                self.global_orders_num = []
            self.con_long_idle = None

            # finished transitions
            self.state_grid_array_done = np.array([])
            self.state_time_array_done = np.array([])
            self.action_array_done = np.array([])
            self.next_state_grid_array_done = np.array([])
            self.next_state_time_array_done = np.array([])
            self.reward_array_done = np.array([])
            self.done_array = np.array([])

         
            # rl for repositioning

        # construct order tabledd
        # TJ
        if self.experiment_date in self.request_all.keys():
            self.request_databases = deepcopy(self.request_all[self.experiment_date])  
        else:
            self.request_databases = []
        # TJ

        request_list = []
        for i in range(env_params['t_initial'],env_params['t_end']):
            request_list.extend(self.request_databases[i])
        column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                       'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                       'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
        self.end_of_episode = 0  # rl for matching
        self.requests = pd.DataFrame(request_list,columns=column_name)
        self.requests['designed_reward'] =  2.5 + 0.5 * ((1000 * self.requests['trip_distance'] - 322).clip(lower=0) / 322) 
        self.requests['trip_time']  = self.requests['trip_distance'] / self.vehicle_speed * 3600
        self.requests['matching_time'] = 0
        self.requests['pickup_end_time'] = 0
        self.requests['delivery_end_time'] = 0
        # TJ
        self.wait_requests = pd.DataFrame(columns=self.request_columns)
        self.matched_requests = pd.DataFrame(columns=self.request_columns)
        # TJ
        self.total_reward = 0
        self.cancelled_penalty = 0
        self.cumulative_on_trip_driver_num = 0
        self.occupancy_rate = 0
        self.total_service_time = 0
        self.total_pickup_time = 0
        self.occupancy_rate_no_pickup = 0
        self.total_online_time = self.driver_table.shape[0] * (self.t_end - self.t_initial)
        self.waiting_time = 0
        self.pickup_time = 0

        self.matched_requests_num = 0.0000001

        self.transfer_request_num = 0
        self.long_requests_num = 0.0000001
        self.medium_requests_num = 0.0000001
        self.short_requests_num = 0.0000001
        self.total_request_num = 0.0000001
        self.time_step1 = 0
        self.time_step2 = 0
        self.step3 = 0
        self.step4 = 0
        self.step4_1 = 0
        self.step5 = 0
        self.step6 = 0
        self.step7 = 0
        # JL
        self.total_pickup_dist = 0
        self.total_reward_per_pickup_dist = 0
        self.idle_drivers_per_grid = None
        self.open_orders_per_grid = None
        self.terminal_orders_with_rewards = {}
        self.orders = OrderTrajectory()
        # JL

    def reset(self):
        self.initial_base_tables()

    def update_info_after_matching_multi_process(self, matched_pair_actual_indexes, matched_itinerary):
        """
        This function used to update driver table and wait requests after matching
        :param matched_pair_actual_indexes: matched pair including driver id and order id
        :param matched_itinerary: including driver pick up route info
        :return: matched requests and wait requests
        """
        new_matched_requests = pd.DataFrame([], columns=self.request_columns)
        update_wait_requests = pd.DataFrame([], columns=self.request_columns)

        matched_pair_index_df = pd.DataFrame(matched_pair_actual_indexes, columns=['order_id', 'driver_id', 'weight', 'pickup_distance'])
        # the index of matched_itinerary_df can be aligned with new_matched_requests
        matched_itinerary_df = pd.DataFrame(columns=['itinerary_node_list', 'itinerary_segment_dis_list', 'pickup_distance'])
        if len(matched_itinerary) > 0:
            matched_itinerary_df['itinerary_node_list'] = matched_itinerary[0]
            matched_itinerary_df['itinerary_segment_dis_list'] = matched_itinerary[1]
            matched_itinerary_df['pickup_distance'] = matched_itinerary[2]
            matched_itinerary_df['pickup_time'] = matched_itinerary_df['pickup_distance'].values / self.vehicle_speed * 3600

        matched_order_id_list = matched_pair_index_df['order_id'].values.tolist()
        # mask for matched orders
        con_matched = self.wait_requests['order_id'].isin(matched_order_id_list)
        # mask for orders still waiting to match
        con_keep_wait = self.wait_requests['wait_time'] <= self.wait_requests['maximum_wait_time']
        # update the `self.wait_requests` table: 1. not matched 2. still waiting to match (matching time tolerance)
        update_wait_requests = pd.concat([update_wait_requests, self.wait_requests[~con_matched & con_keep_wait]], axis=0)
        
        ###### for RL matching radius #######
        # an order will be cancelled due to: 1. the matching time is too long; or 2. the pick-up time is too long
        # for scenario 1 the matching time is too long:
        # mask for orders cancelled during matching
        cancel_due_to_matching = ~con_keep_wait
        # for cancelled orders, negative reward shall be applied as penalty
        if self.experiment_mode == "train" and self.radius_method == 'rl' and cancel_due_to_matching.sum() > 0:
            cancelled_requests = self.wait_requests[cancel_due_to_matching]
            # self.detailed_log.add_entry('cancelled_penalty_matching', self.time, cancelled_requests['origin_id'].values, 
            #                             cancelled_requests['designed_reward'].values)
            self.cancelled_penalty += np.sum(cancelled_requests['designed_reward'].values)
            def compute_rewards(row):
                order_id = row.name  # Since 'order_id' is now the index, use row.name to get it
                reward = -1 * row['designed_reward']
                return self.orders.compute_discounted_rewards(order_id, reward)
            cancelled_requests.set_index('order_id', inplace=True)
            self.terminal_orders_with_rewards = cancelled_requests.apply(compute_rewards, axis=1).to_dict()
        ###### for RL matching radius #######

        # extract the order is matched
        df_matched = self.wait_requests[con_matched].reset_index(drop=True)
        # no matched orders:
        if df_matched.shape[0] == 0: 
            return new_matched_requests, update_wait_requests
        # when there are matched orders
        idle_driver_table = self.driver_table[(self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)]
        order_array = df_matched['order_id'].values
        cor_order = []
        cor_driver = []
        for i in range(len(matched_pair_index_df)):
            cor_order.append(np.argwhere(order_array == matched_pair_index_df['order_id'][i])[0][0])
            cor_driver.append(idle_driver_table[idle_driver_table['driver_id'] == matched_pair_index_df['driver_id'][i]].index[0])
        cor_driver = np.array(cor_driver)
        df_matched = df_matched.iloc[cor_order, :]

        # 1.1 driver decide whether to cancel or not
        # by default, the driver will notr cancel; self.driver_cancel_prob_array can used to manage the cancellation scenario
        driver_cancel_prob = np.zeros(len(matched_pair_index_df))
        prob_array = np.random.rand(len(driver_cancel_prob))
        con_driver_remain = prob_array >= driver_cancel_prob

        # 1.2 customer decided whether to cancel (due to long pickup time)
        # price and pickup time moudle which used to judge whether cancel the order-driver pair
        con_passenger_remain = df_matched['maximum_pickup_time_passenger_can_tolerate'].values >= \
                                                    matched_itinerary_df['pickup_time'].values
        ###### for RL matching radius #######
        cancelled_due_to_pickup = ~con_passenger_remain
        # for cancelled orders, negative reward shall be applied as penalty
        if self.experiment_mode == "train" and self.radius_method == 'rl' and cancelled_due_to_pickup.sum() > 0:
            # self.detailed_log.add_entry('cancelled_penalty_pickup', self.time, df_matched.loc[cancelled_due_to_pickup, 'origin_id'].values, 
            #         df_matched.loc[cancelled_due_to_pickup, 'designed_reward'].values)
            cancelled_requests = df_matched[cancelled_due_to_pickup]
            self.cancelled_penalty += np.sum(cancelled_requests['designed_reward'].values)
            # for RL matching radius
            def compute_rewards(row):
                order_id = row.name  # Since 'order_id' is now the index, use row.name to get it
                reward = -1 * row['designed_reward']
                return self.orders.compute_discounted_rewards(order_id, reward)
            cancelled_requests.set_index('order_id', inplace=True)
            self.terminal_orders_with_rewards = cancelled_requests.apply(compute_rewards, axis=1).to_dict()
        ###### for RL matching radius #######
        
        # mask for not cancelled order-driver pair
        con_remain = con_driver_remain & con_passenger_remain
        # 2.1 driver cancelled by the order
        self.driver_table.loc[cor_driver[~con_remain], ['status', 'remaining_time', 'total_idle_time']] = 0
        # 2.2 order not cancelled
        new_matched_requests = df_matched[con_remain]
        new_matched_requests['t_matched'] = self.time
        new_matched_requests['pickup_distance'] = matched_itinerary_df[con_remain]['pickup_distance'].values
        new_matched_requests['pickup_time'] = new_matched_requests['pickup_distance'].values / self.vehicle_speed * 3600
        new_matched_requests['t_end'] = self.time + new_matched_requests['pickup_time'].values + new_matched_requests['trip_time'].values
        new_matched_requests['status'] = 1
        new_matched_requests['driver_id'] = matched_pair_index_df[con_remain]['driver_id'].values

        ###### for RL matching radius #######
        if self.experiment_mode == "train" and self.radius_method == 'rl' and new_matched_requests.shape[0] > 0:
            matched_requests = deepcopy(new_matched_requests)
            # self.detailed_log.add_entry("designed_reward", self.time, matched_requests['origin_id'].values, 
            #                             matched_requests['designed_reward'].values)
            def compute_rewards(row):
                order_id = row.name  # Since 'order_id' is now the index, use row.name to get it
                reward = row['designed_reward']
                return self.orders.compute_discounted_rewards(order_id, reward)
            # Set 'grid_id' as the index
            matched_requests.set_index('order_id', inplace=True)
            self.terminal_orders_with_rewards = matched_requests.apply(compute_rewards, axis=1).to_dict()
        ###### for RL matching radius #######

        ######## log ########
        self.pickup_time += np.sum(new_matched_requests['pickup_time'].values)
        pickup_time_by_grid = new_matched_requests.groupby('origin_id')['pickup_time'].mean()

        self.total_service_time += np.sum(new_matched_requests['trip_time'].values)
        extra_time = new_matched_requests['t_end'].values - self.t_end
        extra_time[extra_time < 0] = 0
        self.total_service_time -= np.sum(extra_time)
        self.occupancy_rate_no_pickup = self.total_service_time / self.total_online_time 

        self.total_pickup_time += np.sum(new_matched_requests['pickup_time'].values) 
        extra_time = self.time + new_matched_requests['pickup_time'].values - self.t_end
        extra_time[extra_time < 0] = 0
        self.total_pickup_time -= np.sum(extra_time)
        self.occupancy_rate = (self.total_pickup_time + self.total_service_time) / self.total_online_time  

        # self.detailed_log.add_entry("designed_reward", self.time, new_matched_requests['origin_id'].values, 
        #                             new_matched_requests['designed_reward'].values)
        # self.detailed_log.add_entry("pickup_time", self.time, new_matched_requests['origin_id'].values, 
        #                             new_matched_requests['pickup_time'].values)
        # self.detailed_log.add_entry("delivery_time", self.time, new_matched_requests['origin_id'].values, 
        #                             new_matched_requests['trip_time'].values)
        ######## log ########
        
        # 2.3 driver not cancelled
        self.driver_table.loc[cor_driver[con_remain], 'status'] = 2
        self.driver_table.loc[cor_driver[con_remain], 'target_loc_lng'] = new_matched_requests['dest_lng'].values
        self.driver_table.loc[cor_driver[con_remain], 'target_loc_lat'] = new_matched_requests['dest_lat'].values
        self.driver_table.loc[cor_driver[con_remain], 'target_grid_id'] = new_matched_requests['dest_grid_id'].values
        self.driver_table.loc[cor_driver[con_remain], 'remaining_time'] = new_matched_requests['pickup_time'].values
        self.driver_table.loc[cor_driver[con_remain], 'matched_order_id'] = new_matched_requests['order_id'].values
        self.driver_table.loc[cor_driver[con_remain], 'total_idle_time'] = 0
        self.driver_table.loc[cor_driver[con_remain], 'time_to_last_cruising'] = 0
        self.driver_table.loc[cor_driver[con_remain], 'current_road_node_index'] = 0
        # itinerary: pick up + delivery
        self.driver_table.loc[cor_driver[con_remain], 'itinerary_node_list'] = \
            (matched_itinerary_df[con_remain]['itinerary_node_list'] + new_matched_requests['itinerary_node_list']).values
        self.driver_table.loc[cor_driver[con_remain], 'itinerary_segment_dis_list'] = \
            (matched_itinerary_df[con_remain]['itinerary_segment_dis_list'] + new_matched_requests['itinerary_segment_dis_list']).values
        self.driver_table.loc[cor_driver[con_remain], 'remaining_time_for_current_node'] = \
            matched_itinerary_df[con_remain]['itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600

        return new_matched_requests, update_wait_requests

    def step_bootstrap_new_orders(self):
        """
        This function used to generate initial order by different time
        :return:
        """
        # directly sample orders from the historical order database
        sampled_requests = []
        temp_request = []
        # TJ 当更换为按照日期训练时 进行调整
        min_time = max(env_params['t_initial'], self.time - self.request_interval)
        for time in range(min_time, self.time):
            temp_request.extend(self.request_databases[time])
    
        if temp_request == []:
            return
        
        sampled_requests = []
        if self.order_sample_ratio == 1.0:
            sampled_requests = temp_request
        else:
            # sample a portion of historical orders
            database_size = len(temp_request)
            num_request = int(np.rint(self.order_sample_ratio * database_size))
            if num_request < database_size:
                np.random.seed(42)
                sampled_request_index = np.random.choice(database_size, num_request, replace=False).tolist()
                sampled_requests = [temp_request[index] for index in sampled_request_index]
            
        # generate complete information for new orders
        column_name = ['order_id', 'origin_id', 'origin_lat', 'origin_lng', 'dest_id', 'dest_lat', 'dest_lng',
                        'trip_distance', 'start_time', 'origin_grid_id', 'dest_grid_id', 'itinerary_node_list',
                        'itinerary_segment_dis_list', 'trip_time', 'designed_reward', 'cancel_prob']
        
        if len(sampled_requests) > 0:
            wait_info = pd.DataFrame(sampled_requests, columns=column_name)
            sampled_requests_array = np.array(sampled_requests)
            wait_info['itinerary_node_list'] = list(map(lambda x: x[0], sampled_requests_array[:, 11]))
            wait_info['itinerary_segment_dis_list'] = list(map(lambda x: x[0], sampled_requests_array[:, 12]))
            wait_info['start_time'] = self.time
            wait_info['trip_distance'] = np.array(sampled_requests)[:, 7]
            wait_info['trip_time'] = wait_info['trip_distance'] / self.vehicle_speed * 3600
            wait_info['designed_reward'] = 2.5 + 0.5 * (
                                        (wait_info['trip_distance'] * 1000 - 322).clip(lower=0) / 322
                                        ).astype(int) 

                # rl for matching
            wait_info['wait_time'] = 0
            wait_info['status'] = 0
            wait_info['maximum_wait_time'] = self.maximum_wait_time_mean
            wait_info['weight'] = wait_info['designed_reward'] 

            # passenger maximum tolerable price & pickup time
            wait_info['maximum_price_passenger_can_tolerate'] = np.full(len(wait_info), env_params['maximum_price_passenger_can_tolerate_mean'])
            wait_info = wait_info[wait_info['maximum_price_passenger_can_tolerate'] >= wait_info['trip_distance'] * env_params['price_per_km']]
            wait_info['maximum_pickup_time_passenger_can_tolerate'] = np.full(len(wait_info), env_params['maximum_pickup_time_passenger_can_tolerate_mean'])
            
            self.wait_requests = pd.concat([self.wait_requests, wait_info], ignore_index=True)

            # statistics
            self.total_request_num += wait_info.shape[0]
        return

    def cruise_and_reposition(self):
        """
        This function used to judge the drivers' status and
         drivers' table
        :return: None
        """
        self.driver_columns = ['driver_id', 'start_time', 'end_time', 'lng', 'lat', 'grid_id', 'status',
                               'target_loc_lng', 'target_loc_lat', 'target_grid_id', 'remaining_time',
                               'matched_order_id', 'total_idle_time', 'time_to_last_cruising', 'current_road_node_index',
                               'remaining_time_for_current_node', 'itinerary_node_list', 'itinerary_segment_dis_list']

        # reposition decision
        # total_idle_time 为reposition间的间隔， time to last both-rg-cruising 为cruising间的间隔。
        if self.reposition_flag and (self.rl_mode == "matching" or self.rl_mode == "na"):
            con_eligibe = (self.driver_table['total_idle_time'] > self.eligible_time_for_reposition) & \
                          (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            eligible_driver_index = np.array(eligible_driver_table.index)
            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    reposition(eligible_driver_table, self.reposition_mode)
                self.driver_table.loc[eligible_driver_index, 'status'] = 4
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
                self.driver_table.loc[eligible_driver_index, 'total_idle_time'] = 0
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(lambda x: x[-1]).values
                lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(target_node_array)
                self.driver_table.loc[eligible_driver_index, 'target_loc_lng'] = lng_array
                self.driver_table.loc[eligible_driver_index, 'target_loc_lat'] = lat_array
                self.driver_table.loc[eligible_driver_index, 'target_grid_id'] = grid_id_array
        if self.cruise_flag:
            if self.rl_mode in ["na", "matching"]:
                con_eligibe = (self.driver_table['time_to_last_cruising'] >= self.max_idle_time) & \
                            (self.driver_table['status'] == 0)
            elif self.rl_mode == 'reposition':
                con_eligibe = (self.driver_table['total_idle_time'] > self.eligible_time_for_reposition) & \
                           (self.driver_table['status'] == 0)
            eligible_driver_table = self.driver_table[con_eligibe]
            eligible_driver_index = list(eligible_driver_table.index)
            if len(eligible_driver_index) > 0:
                itinerary_node_list, itinerary_segment_dis_list, dis_array = \
                    cruising(eligible_driver_table,self.cruise_mode)
                self.driver_table.loc[eligible_driver_index, 'remaining_time'] = dis_array / self.vehicle_speed * 3600
                self.driver_table.loc[eligible_driver_index, 'time_to_last_cruising'] = 0
                self.driver_table.loc[eligible_driver_index, 'current_road_node_index'] = 0
                self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
                self.driver_table.loc[eligible_driver_index, 'remaining_time_for_current_node'] = \
                    self.driver_table.loc[eligible_driver_index, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600

                # TJ
                # origin node
                origin_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(
                    lambda x: x[0]).values
                lng_array,lat_array,grid_id_array = self.RN.get_information_for_nodes(origin_node_array)
                # target node
                target_node_array = self.driver_table.loc[eligible_driver_index, 'itinerary_node_list'].map(
                    lambda x: x[-1]).values
                target_lng_array, target_lat_array, target_grid_array = self.RN.get_information_for_nodes(target_node_array)
                # TJ

                # TJ
                state_array = np.vstack(
                    [self.time + self.delta_t - self.max_idle_time + np.zeros(grid_id_array.shape[0]),
                     grid_id_array]).T
                if self.rl_mode in ["na", "matching"]:
                    remaining_time_array = self.driver_table.loc[eligible_driver_index, 'remaining_time'].values
                elif self.rl_mode == 'reposition':
                    remaining_time_array = self.driver_table.loc[eligible_driver_index, 'remaining_time'].map(
                    lambda x: x[0]).values
                # TJ

                # rl for matching
                # generate idle transition r1
                action_array = np.ones(grid_id_array.shape[0]) + 1

                # TJ
                # next_state_array = np.vstack([self.time + self.delta_t + np.zeros(grid_id_array.shape[0]),
                #                               target_grid_array]).T

                next_state_array = np.vstack([self.time + remaining_time_array,
                                              target_grid_array]).T

                # TJ
                # rl for matching
                self.driver_table.loc[eligible_driver_index, 'target_loc_lng'] = target_lng_array
                self.driver_table.loc[eligible_driver_index, 'target_loc_lat'] = target_lat_array
                self.driver_table.loc[eligible_driver_index, 'target_grid_id'] = target_grid_array

    def real_time_track_recording(self):
        """
        This function used to record the drivers' info which doesn't delivery passengers
        :return: None
        """
        con_real_time = (self.driver_table['status'] == 0) | (self.driver_table['status'] == 3) | \
                        (self.driver_table['status'] == 4)
        real_time_driver_table = self.driver_table.loc[con_real_time, ['driver_id', 'lng', 'lat', 'status']]
        real_time_driver_table['time'] = self.time
        lat_array = real_time_driver_table['lat'].values.tolist()
        lng_array = real_time_driver_table['lng'].values.tolist()
        node_list = []
        grid_list = []
        for i in range(len(lng_array)):
            id = node_coord_to_id[(lng_array[i], lat_array[i])]
            node_list.append(id)
            grid_list.append(result[result['node_id'] == id ]['grid_id'].tolist()[0])
        real_time_driver_table['node_id'] = node_list
        real_time_driver_table['grid_id'] = grid_list
        real_time_driver_table = real_time_driver_table[['driver_id','lat','lng','node_id','grid_id','status','time']]
        real_time_tracks = real_time_driver_table.set_index('driver_id').T.to_dict('list')
        self.new_tracks = {**self.new_tracks, **real_time_tracks}

    # rl for repositioning
    def generate_repo_driver_state(self):
        con_idle = self.driver_table['status'] == 0
        con_long_idle = con_idle & (self.driver_table['total_idle_time'] >= self.max_idle_time)

        # personal state
        new_repo_grid_array = self.driver_table.loc[con_long_idle, 'grid_id'].values
        new_time_array = np.zeros(new_repo_grid_array.shape[0]) + self.time
        self.state_grid_array = np.concatenate([self.state_grid_array, new_repo_grid_array])
        self.state_time_array = np.concatenate([self.state_time_array, new_time_array])

        idle_drivers_by_grid = 0
        waiting_orders_by_grid = 0
        if self.reposition_method == 'A2C':
            # record average idle vehicles and waiting requests in each grid
            # grid_id_idle_drivers = self.driver_table.loc[
            #                con_idle | (self.driver_table['status'] == 2), 'grid_id'].values
            # TJ
            grid_id_idle_drivers = self.driver_table.loc[
                con_idle | (self.driver_table['status'] == 4), 'grid_id'].values
            # TJ
            indices = np.where(grid_id_idle_drivers.reshape(grid_id_idle_drivers.size, 1) == self.zone_id_array)[1]
            kd = np.bincount(indices)
            idle_drivers_by_grid = np.zeros(NUMBER_OF_GRIDS)
            idle_drivers_by_grid[:len(kd)] = kd

            grid_id_wait_orders = self.wait_requests['origin_grid_id'].values
            indices = np.where(grid_id_wait_orders.reshape(grid_id_wait_orders.size, 1) == self.zone_id_array)[1]
            ko = np.bincount(indices)
            waiting_orders_by_grid = np.zeros(NUMBER_OF_GRIDS)
            waiting_orders_by_grid[:len(ko)] = ko

            # global state
            self.global_time.append(self.time)
            self.global_drivers_num.append(idle_drivers_by_grid)
            self.global_orders_num.append(waiting_orders_by_grid)

        self.con_long_idle = con_long_idle
        return [new_repo_grid_array, new_time_array, idle_drivers_by_grid, waiting_orders_by_grid]

    # rl for repositioning


    # rl for repositioning
    def update_repositioning_driver_status(self, action_array):
        # update status of the drivers to be repositioned
        if len(action_array) > 0:
            con_long_idle = self.con_long_idle
            grid_id_array = self.driver_table.loc[con_long_idle, 'grid_id'].values
            indices = np.where(grid_id_array.reshape(grid_id_array.size, 1) == self.zone_id_array)[1]
            all_directions = df_neighbor_centroid.iloc[:, 3:].values
            dest_grid_id_array = all_directions[indices, action_array]

            indices = np.where(dest_grid_id_array.reshape(dest_grid_id_array.size, 1) == self.zone_id_array)[1]
   
            target_lng_lat_array = np.array(df_neighbor_centroid.iloc[indices, 1:3])
            current_lng_lat_array = np.array(self.driver_table.loc[con_long_idle, ['lng', 'lat']].values.tolist())
            itinerary_node_list, itinerary_segment_dis_list, repo_distance_array = route_generation_array(current_lng_lat_array,target_lng_lat_array)
            repo_time_array = repo_distance_array / self.vehicle_speed * 3600
            # self.driver_table.loc[con_long_idle, 'status'] = 2  # status 2 represents the repositioning status
            ####

            ####
            self.driver_table.loc[con_long_idle, 'status'] = 4  # status 4 represents the repositioning status
            self.driver_table.loc[con_long_idle, ['target_loc_lng', 'target_loc_lat']] = target_lng_lat_array
            self.driver_table.loc[con_long_idle, 'target_grid_id'] = dest_grid_id_array
            self.driver_table.loc[con_long_idle, 'remaining_time'] = repo_time_array
            self.driver_table.loc[con_long_idle, 'total_idle_time'] = 0
            self.driver_table.loc[con_long_idle, 'time_to_last_cruising'] = 0
            self.driver_table.loc[con_long_idle, 'current_road_node_index'] = 0
            self.driver_table.loc[con_long_idle, 'itinerary_node_list'] = np.array(itinerary_node_list + [[]], dtype=object)[:-1]
            self.driver_table.loc[con_long_idle, 'itinerary_segment_dis_list'] = np.array(itinerary_segment_dis_list + [[]], dtype=object)[:-1]
            self.driver_table.loc[con_long_idle, 'remaining_time_for_current_node'] = \
            self.driver_table.loc[con_long_idle, 'itinerary_segment_dis_list'].map(lambda x: x[0]).values / self.vehicle_speed * 3600

        # update final transition records
        con_next_state_done = (self.next_state_time_array >= self.time) & (
                    self.next_state_time_array < (self.time + self.delta_t))
        # print("mext time array",self.next_state_time_array)
        # print("con next state done",con_next_state_done)
        if np.any(con_next_state_done):
            num_action = len(action_array)
            if num_action > 0:
                state_time_array_pre = self.state_time_array[:-num_action]
                state_grid_array_pre = self.state_grid_array[:-num_action]
            else:
                state_time_array_pre = self.state_time_array
                state_grid_array_pre = self.state_grid_array
            self.state_time_array_done = np.concatenate([self.state_time_array_done,
                                                         state_time_array_pre[con_next_state_done]])
            self.state_grid_array_done = np.concatenate([self.state_grid_array_done,
                                                         state_grid_array_pre[con_next_state_done]])
            self.action_array_done = np.concatenate(
                [self.action_array_done, self.action_array[con_next_state_done]])
            self.next_state_time_array_done = np.concatenate([self.next_state_time_array_done,
                                                              self.next_state_time_array[con_next_state_done]])
            self.next_state_grid_array_done = np.concatenate([self.next_state_grid_array_done,
                                                              self.next_state_grid_array[con_next_state_done]])
            next_grid_id_array = self.next_state_grid_array[con_next_state_done]
            indices = np.where(next_grid_id_array.reshape(next_grid_id_array.size, 1) == self.zone_id_array)[1]
            self.reward_array_done = np.concatenate([self.reward_array_done, self.avg_revenue_by_grid[indices]])
            if num_action > 0:
                self.state_time_array = np.concatenate(
                    [state_time_array_pre[~con_next_state_done], self.state_time_array[-num_action:]])
                self.state_grid_array = np.concatenate(
                    [state_grid_array_pre[~con_next_state_done], self.state_grid_array[-num_action:]])
            else:
                self.state_time_array = state_time_array_pre[~con_next_state_done]
                self.state_grid_array = state_grid_array_pre[~con_next_state_done]
            self.action_array = self.action_array[~con_next_state_done]
            self.next_state_time_array = self.next_state_time_array[~con_next_state_done]
            self.next_state_grid_array = self.next_state_grid_array[~con_next_state_done]

        # update temporary transition records
        if len(action_array) > 0:
            self.action_array = np.concatenate([self.action_array, action_array])
            self.next_state_time_array = np.concatenate([self.next_state_time_array, repo_time_array + self.time])
            self.next_state_grid_array = np.concatenate([self.next_state_grid_array, dest_grid_id_array])

    # rl for repositioning

    def update_state(self):
        """
        This function used to update the drivers' status and info
        :return: None
        """
        # update next state
        # status ：0 cruise (park or cruise)， 1 delivery，2 pickup, 3 offline, 4 reposition
        self.driver_table['current_road_node_index'] = self.driver_table['current_road_node_index'].values.astype(int)

        loc_cruise = self.driver_table['status'] == 0  
        loc_reposition = self.driver_table['status'] == 4
        loc_parking = loc_cruise & (self.driver_table['remaining_time'] == 0)
        loc_actually_cruising = (loc_cruise | loc_reposition) & (self.driver_table['remaining_time'] > 0)
        self.driver_table['remaining_time'] = self.driver_table['remaining_time'].values - self.delta_t
        loc_finished = self.driver_table['remaining_time'] <= 0
        loc_unfinished = ~loc_finished
        loc_delivery = self.driver_table['status'] == 1
        loc_pickup = self.driver_table['status'] == 2
        loc_reposition = self.driver_table['status'] == 4
        loc_road_node_transfer = self.driver_table['remaining_time_for_current_node'].values - self.delta_t <= 0

        for order_id,remaining_time in self.driver_table.loc[loc_finished & loc_pickup, ['matched_order_id','remaining_time']].values.tolist():
            # print(order_id)

            self.requests.loc[self.requests['order_id'] == order_id,'pickup_end_time'] = self.time + remaining_time + env_params['delta_t']

        for order_id,remaining_time in self.driver_table.loc[loc_finished & loc_delivery, ['matched_order_id','remaining_time']].values.tolist():
            self.requests.loc[self.requests['order_id'] == order_id,'delivery_end_time'] = self.time + remaining_time + env_params['delta_t']



        # for unfinished tasks
        self.driver_table.loc[loc_cruise, 'total_idle_time'] += self.delta_t
        con_real_time_ongoing = loc_unfinished & (loc_cruise | loc_reposition | loc_delivery) | loc_pickup 
        self.driver_table.loc[~loc_road_node_transfer & con_real_time_ongoing, 'remaining_time_for_current_node'] -= self.delta_t

        road_node_transfer_list = list(self.driver_table[loc_road_node_transfer & con_real_time_ongoing].index)
        current_road_node_index_array = self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'].values
        current_remaining_time_for_node_array = self.driver_table.loc[road_node_transfer_list, 'remaining_time_for_current_node'].values
        transfer_itinerary_node_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_node_list'].values
        transfer_itinerary_segment_dis_list = self.driver_table.loc[road_node_transfer_list, 'itinerary_segment_dis_list'].values
        new_road_node_index_array = np.zeros(len(road_node_transfer_list))
        new_road_node_array = np.zeros(new_road_node_index_array.shape[0])
        new_remaining_time_for_node_array = np.zeros(new_road_node_index_array.shape[0])

        # update the driver itinerary list
        for i in range(len(road_node_transfer_list)):
            current_node_index = current_road_node_index_array[i]
            itinerary_segment_time = np.array(transfer_itinerary_segment_dis_list[i][current_node_index:]) / self.vehicle_speed * 3600
            itinerary_segment_time[0] = current_remaining_time_for_node_array[i]
            itinerary_segment_cumsum_time = itinerary_segment_time.cumsum()
            new_road_node_index = (itinerary_segment_cumsum_time > self.delta_t).argmax()
            new_remaining_time = itinerary_segment_cumsum_time[new_road_node_index] - self.delta_t
            if itinerary_segment_cumsum_time[-1] <= self.delta_t:
                new_road_node_index = len(transfer_itinerary_segment_dis_list[i])-1
            else:
                new_road_node_index = new_road_node_index + current_node_index
            new_road_node = transfer_itinerary_node_list[i][new_road_node_index]

            new_road_node_index_array[i] = new_road_node_index
            new_road_node_array[i] = new_road_node
            new_remaining_time_for_node_array[i] = new_remaining_time

        self.driver_table.loc[road_node_transfer_list, 'current_road_node_index'] = new_road_node_index_array.astype(int)
        self.driver_table.loc[road_node_transfer_list, 'remaining_time_for_current_node'] = new_remaining_time_for_node_array
        lng_array, lat_array, grid_id_array = self.RN.get_information_for_nodes(new_road_node_array)
        self.driver_table.loc[road_node_transfer_list, 'lng'] = lng_array
        self.driver_table.loc[road_node_transfer_list, 'lat'] = lat_array
        self.driver_table.loc[road_node_transfer_list, 'grid_id'] = grid_id_array

        # for all the finished tasks
        self.driver_table.loc[loc_finished & (~ loc_pickup), 'remaining_time'] = 0
        con_not_pickup = loc_finished & (loc_actually_cruising | loc_delivery | loc_reposition)
        con_not_pickup_actually_cruising = loc_finished & (loc_delivery | loc_reposition)
        self.driver_table.loc[con_not_pickup, 'lng'] = self.driver_table.loc[con_not_pickup, 'target_loc_lng'].values
        self.driver_table.loc[con_not_pickup, 'lat'] = self.driver_table.loc[con_not_pickup, 'target_loc_lat'].values
        self.driver_table.loc[con_not_pickup, 'grid_id'] = self.driver_table.loc[con_not_pickup, 'target_grid_id'].values
        self.driver_table.loc[con_not_pickup, ['status', 'time_to_last_cruising', 'current_road_node_index',
                                               'remaining_time_for_current_node']] = 0
        self.driver_table.loc[con_not_pickup_actually_cruising, 'total_idle_time'] = 0
        shape = self.driver_table[con_not_pickup].shape[0]
        empty_list = [[] for _ in range(shape)]
        self.driver_table.loc[con_not_pickup, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
        self.driver_table.loc[con_not_pickup, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]

        # for parking finished
        self.driver_table.loc[loc_parking, 'time_to_last_cruising'] += self.delta_t

        # for delivery finished
        self.driver_table.loc[loc_finished & loc_delivery, 'matched_order_id'] = 'None'


        # self.driver_table.loc[loc_finished & loc_delivery]
        """
        for pickup    delivery是载客  pickup是接客
        分两种情况，一种是下一时刻pickup 和 delivery都完成，另一种是下一时刻pickup 完成，delivery没完成
        当前版本delivery直接跳转，因此不需要做更新其中间路线的处理。车辆在pickup完成后，delivery完成前都实际处在pickup location。完成任务后直接跳转到destination
        如果需要考虑delivery的中间路线，可以把pickup和delivery状态进行融合
        """

        finished_pickup_driver_index_array = np.array(self.driver_table[loc_finished & loc_pickup].index)
        current_road_node_index_array = self.driver_table.loc[finished_pickup_driver_index_array,
                                                              'current_road_node_index'].values
        itinerary_segment_dis_list = self.driver_table.loc[finished_pickup_driver_index_array,
                                                           'itinerary_segment_dis_list'].values
        remaining_time_current_node_temp = self.driver_table.loc[finished_pickup_driver_index_array,
                                                           'remaining_time_for_current_node'].values

        # load pickup time
        remaining_time_array = np.zeros(len(finished_pickup_driver_index_array))
        for i in range(remaining_time_array.shape[0]):
            current_node_index = current_road_node_index_array[i]
            remaining_time_array[i] = np.sum(itinerary_segment_dis_list[i][current_node_index+1:]) / self.vehicle_speed * 3600 + remaining_time_current_node_temp[i]
        delivery_not_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array > 0]
        delivery_finished_driver_index = finished_pickup_driver_index_array[remaining_time_array <= 0]
        self.driver_table.loc[delivery_not_finished_driver_index, 'status'] = 1
        self.driver_table.loc[delivery_not_finished_driver_index, 'remaining_time'] = remaining_time_array[remaining_time_array > 0]
        if len(delivery_finished_driver_index > 0):
            self.driver_table.loc[delivery_finished_driver_index, 'lng'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_loc_lng'].values
            self.driver_table.loc[delivery_finished_driver_index, 'lat'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_loc_lat'].values
            self.driver_table.loc[delivery_finished_driver_index, 'grid_id'] = \
                self.driver_table.loc[delivery_finished_driver_index, 'target_grid_id'].values
            self.driver_table.loc[delivery_finished_driver_index, ['status', 'time_to_last_cruising',
                                                                   'current_road_node_index',
                                                                   'remaining_time_for_current_node']] = 0
            self.driver_table.loc[delivery_finished_driver_index, 'total_idle_time'] = 0
            shape = self.driver_table.loc[delivery_finished_driver_index].values.shape[0]
            empty_list = [[] for _ in range(shape)]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_node_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'itinerary_segment_dis_list'] = np.array(empty_list + [[-1]], dtype=object)[:-1]
            self.driver_table.loc[delivery_finished_driver_index, 'matched_order_id'] = 'None'
        self.wait_requests['wait_time'] += self.delta_t

        if self.radius_method == "greedy":
            # eligible_update = self.wait_requests['wait_time'] / self.delta_t % 5 == 0
            # self.wait_requests.loc[eligible_update, 'matching_radius'] = (1 + self.wait_requests['wait_time'] / self.delta_t) * 0.5
            self.wait_requests['matching_radius'] = (1 + self.wait_requests['wait_time'] / self.delta_t) * 0.5
            # maximum radius is 6
            self.wait_requests.loc[self.wait_requests['matching_radius'] > 6.0, 'matching_radius'] = 6.0
            
        return

    def driver_online_offline_update(self):
        """current_step
        update driver online/offline status
        currently, only offline con need to be considered.
        offline driver will be deleted from the table
        :return: None
        """
        next_time = self.time + self.delta_t
        self.driver_table = driver_online_offline_decision(self.driver_table, next_time)
        return

    def update_time(self):
        """
        This function used to count time
        :return:
        """
        self.time += self.delta_t
        self.current_step = int((self.time - self.t_initial) // self.delta_t)

        # rl for matching
        if self.current_step >= self.finish_run_step:
            self.end_of_episode = 1
        # rl for matching
        return
    
    def step(self):
        """
        This function used to run the simulator step by step
        """
        # Step 1: order dispatching
        ####### log ################
        # orders_by_grid = self.wait_requests.groupby('origin_id').size()
        # self.detailed_log.add_entry('order_num', self.time, np.array(orders_by_grid.index), np.array(orders_by_grid.values))
        # drivers_by_grid = self.driver_table.groupby('grid_id').size()
        # self.detailed_log.add_entry('driver_num', self.time, np.array(drivers_by_grid.index), np.array(drivers_by_grid.values))
        ####### log ################
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        if self.radius_method in["rl", "greedy"]:
            matched_pair_actual_indexes, matched_itinerary = order_dispatch_radius(wait_requests, driver_table, 
                                           self.dispatch_method, self.method, self.adjust_reward_by_radius)
        elif self.radius_method == "fixed":
            matched_pair_actual_indexes, matched_itinerary = order_dispatch(wait_requests, driver_table,
                                                            self.maximal_pickup_distance,
                                                            self.dispatch_method,self.method)
        
        
        self.waiting_time += np.sum(self.wait_requests['wait_time'].values)
        wait_time_by_grid = self.wait_requests.groupby('origin_id')['wait_time'].mean()
        # self.detailed_log.add_entry('pickup_time', self.time, np.array(wait_time_by_grid.index), np.array(wait_time_by_grid.values))
            
        # Step 2: driver/passenger reaction after dispatching
        df_new_matched_requests, df_update_wait_requests = \
            self.update_info_after_matching_multi_process(matched_pair_actual_indexes, matched_itinerary)
        
        self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
        self.matched_requests = self.matched_requests.reset_index(drop=True)
        self.wait_requests = df_update_wait_requests.reset_index(drop=True)
        if isinstance(self.record, str):
            self.record = df_new_matched_requests
        else:
            self.record = pd.concat([self.record, df_new_matched_requests], axis=0, ignore_index=True)

        ####### log ################
        if len(df_new_matched_requests) != 0:
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
            self.matched_requests_num += len(df_new_matched_requests)
        ####### log ################
        
        # Step 3: bootstrap new orders
        self.step_bootstrap_new_orders()

        # Step 4: both-rg-cruising and/or repositioning decision
        # self.cruise_and_reposition()

        # Step 5: update next state for drivers
        self.update_state()

        # Step 6： online/offline update()
        self.driver_online_offline_update()

        # Step 7: update time
        self.update_time()
      
    def step1(self): # rl for repositioning
        # Step 1: bipartite matching
        wait_requests = deepcopy(self.wait_requests)
        driver_table = deepcopy(self.driver_table)
        matched_pair_actual_indexes,matched_itinerary = order_dispatch(wait_requests, driver_table, self.maximal_pickup_distance ,self.dispatch_method )

        # Step 
        # removed matched requests from wait list, and put them into matched list
        # delete orders with too much waiting time
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 1].shape[0]
        self.cumulative_on_trip_driver_num += self.driver_table[self.driver_table['status'] == 2].shape[0]
        self.occupancy_rate = self.cumulative_on_trip_driver_num / (
                    (1 + self.current_step) * self.driver_table.shape[0])
        df_new_matched_requests, df_update_wait_requests = self.update_info_after_matching_multi_process(matched_pair_actual_indexes,matched_itinerary)
        if self.rl_mode == 'reposition' and self.end_of_episode == 0:
            self.matched_requests = pd.concat([self.matched_requests, df_new_matched_requests], axis=0)
            self.matched_requests = self.matched_requests.reset_index(drop=True)
            self.wait_requests = df_update_wait_requests.reset_index(drop=True)
        self.matched_requests_num += len(df_new_matched_requests)
        # TJ
        if len(df_new_matched_requests) != 0:
            self.total_reward += np.sum(df_new_matched_requests['designed_reward'].values)
        else:
            self.total_reward += 0
        # TJ
        # Step 3: generate new orders
        self.step_bootstrap_new_orders()

        # Step 4: generate states of the repositioning drivers
        state_list = self.generate_repo_driver_state()   # rl for repositioning

        return state_list   # rl for repositioning

    def step2(self, action_array):  # rl for repositioning
        # Step 5: update repositioning driver status
        action_array = action_array.astype(int)  # rl for repositioning
        self.update_repositioning_driver_status(action_array)  # rl for repositioning
        if self.rl_mode == 'matching' and self.method == 'instant_reward_no_subway':
            self.cruise_and_reposition()
        # Step 6: update next state and time
        self.update_state()
        self.update_time()

        return
    
    def get_demand_supply(self):
        # Group by grid_id and count the records in driver dataframe
        idle_drivers = self.driver_table[(self.driver_table['status'] == 0) | (self.driver_table['status'] == 4)]
        self.idle_drivers_per_grid = idle_drivers.groupby('grid_id').size()
        # Group by 'grid_id' in wait_requests to count the num_open_order_per_grid
        self.open_orders_per_grid = self.wait_requests.groupby('origin_id').size()