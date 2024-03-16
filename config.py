env_params = { 
    't_initial': 18000, 
    't_end': 36000, 
    'delta_t' : 60,  # s
    'vehicle_speed' : 22.788,   # km / h
    'repo_speed' : 22.788, # should be the same as 'vehicle_speed'
    'order_sample_ratio' : 1.0, 
    'driver_sample_ratio' : 1.0,
    'maximum_wait_time_mean' : 300, # 5 min
    'maximum_wait_time_std' : 0,
    "maximum_pickup_time_passenger_can_tolerate_mean": 900,  # 900s = 15min
    "maximum_pickup_time_passenger_can_tolerate_std":0, # s
    "maximum_price_passenger_can_tolerate_mean":float('inf'), # ￥
    "maximum_price_passenger_can_tolerate_std":0,  # ￥
    'maximal_pickup_distance' : 1,  # km
    'request_interval': 60,  # s  # should be the same as 'delta_t'
    'cruise_flag' : False,
    'delivery_mode':'rg',
    'pickup_mode':'rg',
    'max_idle_time' : 300, # 5min
    'cruise_mode': 'random',
    'reposition_flag': False,
    'eligible_time_for_reposition' : 300, # 5 min
    'reposition_mode': '',
    'track_recording_flag' : False, # DO NOT use this -> slow down the simulator drastically
    'driver_far_matching_cancel_prob_file' : 'driver_far_matching_cancel_prob',
    'request_file_name' :  'orders_0.6_complete', # 'input/order-11-13-frac=0.1',
    'driver_file_name' : 'driver_100',
    'dispatch_method': 'LD', #LD: lagarange decomposition method designed by Peibo Duan
    'experiment_mode' : 'train',
    'driver_num':100,
    'price_per_km':5,  # ￥ / km
    'road_information_mode':'load',
    'price_increasing_percentage': 0.2,
    'radius_method': 'rl',  # ['rl','fixed', 'greedy']
    'rl_mode': 'na', # ['matching', 'reposition']
    'method': 'sarsa_no_subway',  #  'sarsa_no_subway' / 'pickup_distance' / 'instant_reward_no_subway'   #  rl for matching
    'reposition_method': 'A2C_global_aware',  # A2C, A2C_global_aware, random_cruise, stay  # rl for repositioning
    'pre_trained': False, # load pre-trained parameters 
}

NUM_EPOCH = 120
BATCH_SIZE = 128  # 128/256
TRAIN_DATE_LIST = ['2015-05-04']#, '2015-05-05','2015-05-06', '2015-05-07', '2015-05-08']
TEST_DATE_LIST = ['2015-05-11'] #, '2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15']