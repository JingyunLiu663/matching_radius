env_params = { 
    't_initial': 14400, # 4AM
    't_end':  25200, # 7 AM
    'delta_t' : 5,  # s
    'vehicle_speed' : 22.788,   # km / h
    'repo_speed' : 1, #目前的设定需要与vehicl speed保持一致
    'order_sample_ratio' : 1, 
    'order_generation_mode' : 'sample_from_base',
    'driver_sample_ratio' : 1,
    'maximum_wait_time_mean' : 300,
    'maximum_wait_time_std' : 0,
    "maximum_pickup_time_passenger_can_tolerate_mean":float('inf'),  # s
    "maximum_pickup_time_passenger_can_tolerate_std":0, # s
    "maximum_price_passenger_can_tolerate_mean":float('inf'), # ￥
    "maximum_price_passenger_can_tolerate_std":0,  # ￥
    'maximal_pickup_distance' : 1,  # km
    'request_interval': 5,  # s
    'cruise_flag' :False,
    'delivery_mode':'rg',
    'pickup_mode':'rg',
    'max_idle_time' : 1,
    'cruise_mode': 'random',
    'reposition_flag': False,
    'eligible_time_for_reposition' : 10, # s
    'reposition_mode': '',
    'track_recording_flag' : True,
    'driver_far_matching_cancel_prob_file' : 'driver_far_matching_cancel_prob',
    'input_file_path':'input/dataset.csv',
    'request_file_name' : 'input/order-11-13-frac=0.1', #'toy_requests',
    'driver_file_name' : 'input/driver_info',
    'road_network_file_name' : 'road_network_information.pickle',
    'dispatch_method': 'LD', #LD: lagarange decomposition method designed by Peibo Duan
    'simulator_mode' : 'toy_mode',
    'experiment_mode' : 'train',
    'driver_num':500,
    'side':10, # grid side
    'price_per_km':5,  # ￥ / km
    'road_information_mode':'load',
    'price_increasing_percentage': 0.2,
    'north_lat': 40.8845,
    'south_lat': 40.6968,
    'east_lng': -74.0831,
    'west_lng': -73.8414,
    'rl_mode': 'matching_radius',  # ['reposition','matching_radius', 'random', 'fixed', 'greedy_radius']
    'method': 'sarsa_no_subway',  #  'sarsa_no_subway' / 'pickup_distance' / 'instant_reward_no_subway'   #  rl for matching
    'reposition_method': 'A2C_global_aware',  # A2C, A2C_global_aware, random_cruise, stay  # rl for repositioning
    'dayparting': False, # if true, simulator_env will compute information based on time periods in a day, e.g. 'morning', 'afternoon'
    'pre_trained': False, # load pre-trained parameters #TODO
}

# rl for matching radius
NUM_EPOCH = 120
BATCH_SIZE = 128  # 128/256
# TRAIN_DATE_LIST = ['2015-05-04', '2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08', '2015-05-11', '2015-05-12',
#                    '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
TRAIN_DATE_LIST = ['2015-05-04']#, '2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08']
TEST_DATE_LIST = ['2015-05-11']#, '2015-05-12', '2015-05-13', '2015-05-14', '2015-05-15']
# rl for matching radius