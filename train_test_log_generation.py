import numpy as np
import pickle
'''
The script is used to generate train/test log 
one pickled file is generated
one tensorboard log is generated for in-time monitoring
'''

class TrainTestLog:
    '''
    Use a dictionary to store the train/test metrics for each epoch
    In case more flexible visualization which TensorBoard cannot provide is needed in the future 
    '''
    def __init__(self):
        column_list = ['epoch_average_loss', 'total_adjusted_reward', 'total_reward',
                'total_request_num', 'matched_request_num',
                'matched_request_ratio',
                'waiting_time', 'pickup_time',
                'occupancy_rate','occupancy_rate_no_pickup']
        self.record_dict = {col: [] for col in column_list}

    def save(self, rl_agent, adjust_reward, experiment_mode, rl_mode):
        '''
        rl_agnet: dqn, ddqn, dueling_dqn
        adjust_reward: 0/1
        experiment_mode: "train"/"test"
        rl_mode: "matching_radius", "rl_greedy", "random", "fixed"
        '''
        # serialize the record
        if rl_agent is None:
            if adjust_reward:
                with open(f'{rl_mode}_adjusted_{experiment_mode}.pkl', 'wb') as f:
                    pickle.dump(self.record_dict, f)
            else:
                with open(f'{rl_mode}_immediate_{experiment_mode}.pkl', 'wb') as f:
                    pickle.dump(self.record_dict, f)
        else:
            if adjust_reward:
                with open(f'{rl_mode}_{rl_agent}_adjusted_{experiment_mode}.pkl', 'wb') as f:
                    pickle.dump(self.record_dict, f)
            else:
                with open(f'{rl_mode}_{rl_agent}_immediate_{experiment_mode}.pkl', 'wb') as f:
                    pickle.dump(self.record_dict, f)

class EpochLog:
    '''
    Each epoch consists of len(TRAIN_DATE_LIST)/len(TRAIN_DATE_LIST) data
    '''
    def __init__(self, experiment_mode="train"):
        self.experiment_mode = experiment_mode
        if self.experiment_mode == "train":
            self.episode_time = []
            self.epoch_loss = []

        self.total_reward = []
        self.total_adjusted_reward = []
        self.total_request_num = []
        self.matched_requests_num = []
        self.occupancy_rate = []
        self.occupancy_rate_no_pickup = []
        self.pickup_time = []
        self.waiting_time = []

    def reset(self):
        if self.experiment_mode == "train":
            self.episode_time = []
            self.epoch_loss = []

        self.total_reward = []
        self.total_adjusted_reward = []
        self.total_request_num = []
        self.matched_requests_num = []
        self.occupancy_rate = []
        self.occupancy_rate_no_pickup = []
        self.pickup_time = []
        self.waiting_time = []

    def add_loss_per_learning_step(self, loss):
        self.epoch_loss.append(loss)

    def add_within_epoch(self, reward, adjust_reward, request_num, matched_request_num, 
                 occupancy_rate, occupancy_rate_no_pickup, pickup_time, wait_time, time):
        if self.experiment_mode == "train":
            self.episode_time = time

        self.total_reward.append(reward)
        self.total_adjusted_reward.append(adjust_reward)
        self.total_request_num.append(request_num)
        self.matched_requests_num.append(matched_request_num)
        self.occupancy_rate.append(occupancy_rate)
        self.occupancy_rate_no_pickup.append(occupancy_rate_no_pickup)
        self.pickup_time.append(pickup_time / matched_request_num) # average pickup time per order
        self.waiting_time.append(wait_time / matched_request_num) # average wait time per order


    def add_to_tensorboard(self, agent, epoch):
        if self.experiment_mode == "train":
            agent.train_writer.add_scalar('running time', np.mean(self.episode_time), epoch)
            agent.train_writer.add_scalar('average loss', np.mean(self.epoch_loss), epoch)
            agent.train_writer.add_scalar('total adjusted reward (per pickup distance)', sum(self.total_adjusted_reward), epoch)
            agent.train_writer.add_scalar('total reward', sum(self.total_reward), epoch)
            agent.train_writer.add_scalar('total orders', sum(self.total_request_num), epoch)
            agent.train_writer.add_scalar('matched orders', sum(self.matched_requests_num), epoch)
            agent.train_writer.add_scalar('matched request ratio', sum(self.matched_requests_num)/ sum(self.total_request_num), epoch)
            agent.train_writer.add_scalar('matched occupancy rate', np.mean(self.occupancy_rate), epoch)
            agent.train_writer.add_scalar('matched occupancy rate - no pickup', np.mean(self.occupancy_rate_no_pickup), epoch)
            agent.train_writer.add_scalar('pickup time', np.mean(self.pickup_time), epoch)
            agent.train_writer.add_scalar('waiting time', np.mean(self.waiting_time), epoch)
        else:
            agent.test_writer.add_scalar('total adjusted reward (per pickup distance)', sum(self.total_adjusted_reward), epoch)
            agent.test_writer.add_scalar('total reward', sum(self.total_reward), epoch)
            agent.test_writer.add_scalar('total orders', sum(self.total_request_num), epoch)
            agent.test_writer.add_scalar('matched orders', sum(self.matched_requests_num), epoch)
            agent.test_writer.add_scalar('matched request ratio', sum(self.matched_requests_num)/ sum(self.total_request_num), epoch)
            agent.test_writer.add_scalar('matched occupancy rate', np.mean(self.occupancy_rate), epoch)
            agent.test_writer.add_scalar('matched occupancy rate - no pickup', np.mean(self.occupancy_rate_no_pickup), epoch)
            agent.test_writer.add_scalar('pickup time', np.mean(self.pickup_time), epoch)
            agent.test_writer.add_scalar('waiting time', np.mean(self.waiting_time), epoch)


    def add_to_pickle_file(self, record_dict):
        if self.experiment_mode == "train":
            record_dict['epoch_average_loss'].append(np.mean(self.epoch_loss))
        record_dict['total_adjusted_reward'].append(sum(self.total_adjusted_reward))
        record_dict['total_reward'].append(sum(self.total_reward))
        record_dict['total_request_num'].append(sum(self.total_request_num))
        record_dict['matched_request_num'].append(sum(self.matched_requests_num))
        record_dict['matched_request_ratio'].append(sum(self.matched_requests_num) / sum(self.matched_requests_num))
        record_dict['occupancy_rate'].append(np.mean(self.occupancy_rate))
        record_dict['occupancy_rate_no_pickup'].append(np.mean(self.occupancy_rate_no_pickup))
        record_dict['pickup_time'].append(np.mean(self.pickup_time))
        record_dict['waiting_time'].append(np.mean(self.waiting_time))