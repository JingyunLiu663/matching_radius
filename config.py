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
    'radius_action_space': [0.5, 0.75, 1.0], #list(np.arange(2.0, 6.5, 0.5)),  # Newly Added: [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
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
    # 'method': 'instant_reward_no_subway',
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
    'rl_mode': 'matching_radius',  # ['reposition', 'matching', 'matching_radius']
    'method': 'sarsa_no_subway',  #  'sarsa_no_subway' / 'pickup_distance' / 'instant_reward_no_subway'   #  rl for matching
    'reposition_method': 'A2C_global_aware',  # A2C, A2C_global_aware, random_cruise, stay  # rl for repositioning
    'dayparting': True, # if true, simulator_env will compute information based on time periods in a day, e.g. 'morning', 'afternoon'
    'rl_agent': 'dqn',
    'pre_trained': False, # load pre-trained parameters
}
wait_time_params_dict = {'morning': [2.582, 2.491, 0.026, 1.808, 2.581],
                    'evening': [4.862, 2.485, 0, 1.379, 13.456],
                    'midnight_early': [0, 2.388, 2.972, 2.954, 3.14],
                    'other': [0, 2.017, 2.978, 2.764, 2.973]}

pick_time_params_dict = {'morning': [1.877, 2.018, 2.691, 1.865, 6.683],
                    'evening': [2.673,2.049,2.497,1.736,9.208],
                    'midnight_early': [3.589,2.319,2.185,1.664,9.6],
                    'other': [0,1.886,4.099,3.185,3.636]}

price_params_dict = {'short': [1.245,0.599,10.629,10.305,0.451],
                    'short_medium': [0.451,0.219,19.585,58.407,0.18],
                    'medium_long': [14.411,4.421,11.048,9.228,145],
                    'long': [15.821,3.409,0,16.221,838.587]}

# price_increase_params_dict = {'morning': [0.001,1.181,3.583,4.787,0.001],
#                     'evening': [0,1.21,2.914,5.023,0.013],
#                     'midnight_early': [1.16,0,0,6.366,0],
#                     'other': [0,2.053,0.857,4.666,1.961]}

# rl for matching radius
NUM_EPOCH = 100
UPDATE_INTERVAL = 20
TRAIN_DATE_LIST = ['2015-05-04', '2015-05-05', '2015-05-06', '2015-05-07', '2015-05-08', '2015-05-11', '2015-05-12',
                   '2015-05-13', '2015-05-14', '2015-05-15', '2015-05-18']
TEST_DATE_LIST = ['2015-07-27', '2015-07-28', '2015-07-29', '2015-07-30', '2015-07-31']
# rl for matching radius