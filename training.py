import logging
import time
import os
from dqn import DqnAgent
from ddqn import DDqnAgent
from dueling_dqn import DuelingDqnAgent
from log_generation import ActionTracker
from config import *
from utilities import *
from draw_snapshots import draw_simulation

def update_simulator_args(simulator, args):
    '''
    Adjust the simulator environment based on command line arguments.

    Parameters:
    - simulator (Simulator): An instance of the Simulator class that will be modified.
    - args (Namespace): A namespace object containing command line arguments. Expected attributes include:
      - experiment_mode (str): Mode of the experiment which could be 'train' or 'test'.
      - radius_method (str): Method to determine the pickup radius, options are 'rl' for reinforcement learning or 'fixed' for a predetermined value.
      - adjust_reward (bool): Flag indicating whether to adjust the reward based on radius.
      - radius (float): The maximal pickup distance, used when radius_method is 'fixed'.
      - cruise (int): Flag converted to boolean indicating whether cruising is enabled (1 for enabled).

    Returns:
    None: This function modifies the simulator in-place and does not return any value.
    '''
    simulator.experiment_mode = args.experiment_mode
    simulator.radius_method = args.radius_method
    simulator.rl_agent = args.rl_agent
    simulator.adjust_reward_by_radius = args.adjust_reward
    simulator.maximal_pickup_distance = args.radius # for rl_mode == fixed
 
    if simulator.adjust_reward_by_radius: 
        logging.info("reward adjusted by radius")
    else:
        logging.info("reward not adjusted by radius")

def create_agent(args):
    '''
    Create a Reinforcement Learning agent based on command line arguments.

    Parameters:
    - args (Namespace): A namespace object containing command line arguments. Expected attributes include:
      - rl_agent (str): Selector for the type of RL agent to create ('dqn', 'ddqn', 'dueling_dqn').
      - action_space (int): The size of the action space for the agent.
      - num_layers (int): The number of layers in the neural network.
      - dim_list (list): List of dimensions for each layer in the neural network.
      - lr (float): Learning rate for the agent's optimizer.
      - gamma (float): Discount factor for the agent's future rewards.
      - epsilon (float): Initial value for the epsilon-greedy policy.
      - eps_min (float): Minimum value for epsilon in the epsilon-greedy policy.
      - eps_dec (float): Decrement value for epsilon in the epsilon-greedy policy.
      - target_update (int): Frequency of target network update.
      - experiment_mode (str): Mode of the experiment, such as 'train' or 'test'.
      - adjust_reward (bool): Flag indicating whether to adjust the reward.

    Returns:
    - agent (RLAgent): An instance of the specified RLAgent class.
    '''
    if args.rl_agent == "dqn":
        agent = DqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
        logging.info("dqn agent is created")
    elif args.rl_agent == "ddqn":
        agent = DDqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
        logging.info("double dqn agent is created")
    elif args.rl_agent == "dueling_dqn":
        agent = DuelingDqnAgent(args.action_space, args.num_layers, args.dim_list, 5, args.lr, args.gamma,
                            args.epsilon, args.eps_min, args.eps_dec, args.target_update, args.experiment_mode, args.adjust_reward)
        logging.info("dueling dqn agent is created")
    else:
        raise ValueError(f"RL agent type '{args.rl_agent}' is not recognized.")
    return agent

def load_pretrained_model(agent, args, env_params, model_epoch=0):
    '''
    Load pre-trained model during test stage, or if specified during train stage
    '''
    if env_params['pre_trained'] or args.experiment_mode == 'test':
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        parameter_path = os.path.join(models_dir, f"{args.rl_agent}_epoch{model_epoch}_{args.adjust_reward}_model.pth")
        agent.load_parameters(parameter_path)
        logging.info("pre-trained model is loaded")

def get_states(simulator):
    simulator.get_demand_supply()
    idle_drivers = simulator.idle_drivers_per_grid.reindex(simulator.wait_requests['order_id']).fillna(0)
    open_orders = simulator.open_orders_per_grid.reindex(simulator.wait_requests['order_id']).fillna(0)

    states = list(zip(simulator.wait_requests['order_id'], 
                    [simulator.time]*len(simulator.wait_requests), 
                    idle_drivers, 
                    open_orders, 
                    simulator.wait_requests['wait_time']))
    return states

def get_actions_given_states(agent, states, args):
    action_indices = [agent.choose_action(state) for state in states]
    matching_radius = [args.action_space[i] for i in action_indices]
    return action_indices, matching_radius


def simulation_train(args, agent, simulator, logger, loss_logger):
    models_dir = os.path.join("models", args.rl_agent)
    os.makedirs(models_dir, exist_ok=True)

    for epoch in range(NUM_EPOCH):
        # Initialize replay_buffer
        replay_buffer = ReplayBuffer()

        for date in TRAIN_DATE_LIST:
            # Initialize the environment 
            simulator.experiment_date = date
            # Reset simulator
            simulator.reset()

            for step in range(simulator.finish_run_step):
                losses = []
                if simulator.wait_requests.shape[0] > 0:
                    states = get_states(simulator)
                    action_indices, matching_radius = get_actions_given_states(agent, states, args)
                    # Append the action distribution at time step t to the list
                    simulator.wait_requests['action_index'] = action_indices
                    simulator.wait_requests['matching_radius'] = matching_radius
                
                    # Store state and action in ongoing trajectories
                    # Map order to the trajectory list [s_0, a_0, s_1, a_1, ...]
                    for order_id, state, action_index in zip(simulator.wait_requests['order_id'], states, action_indices):
                        if order_id not in simulator.orders.trajectories:
                            simulator.orders.trajectories[order_id] = []
                        simulator.orders.trajectories[order_id].append((state, action_index))

                # Run the simulator environment
                simulator.step()

                # If any orders have reached a terminal state (traverse the terminal orders dictionary):
                # - Calculate the spread rewards
                # - Add the transitions (along the trajectory) to the replay buffer
                for order_id, rewards in simulator.terminal_orders_with_rewards.items():
                    trajectory = simulator.orders.trajectories.pop(order_id)
                    for i, ((state, action), reward) in enumerate(zip(trajectory, rewards)):
                        if i < len(trajectory) - 1:
                            next_state = trajectory[i+1][0]
                            done = False
                        else:
                            next_state = np.zeros(5)
                            done = True
                        replay_buffer.push((state, action, reward, next_state, done))
                # Clear the terminal orders dictionary
                simulator.terminal_orders_with_rewards = {}

                # If a batch of data is collected, train the deep neural network for 10 times 
                if len(replay_buffer) >= BATCH_SIZE:
                    for i in range(10):
                        states, action_indices, rewards, next_states, done = replay_buffer.sample(BATCH_SIZE)
                        agent.learn(states, action_indices, rewards, next_states, done)
                        losses.append(agent.losses)
            
            loss_logger.info(f"epoch:{epoch}, average_loss: {np.mean(agent.losses)}")
            logger.info(f"epoch: {epoch} == total_reward:{simulator.total_reward},matching_rate:{simulator.matched_requests_num/simulator.total_request_num},total_request_num:{simulator.total_request_num},matched_request_num:{simulator.matched_requests_num},occupancy_rate:{simulator.occupancy_rate},occupancy_rate_no_pickup:{simulator.occupancy_rate_no_pickup},wait_time:{simulator.waiting_time / simulator.matched_requests_num},pickup_time:{simulator.pickup_time / simulator.matched_requests_num}")
        if epoch >= 100:
            parameter_path = os.path.join(models_dir, f"{args.rl_agent}_{epoch}_model.pth")
            agent.save_parameters(parameter_path)

    
def simulation_fixed(simulator, logger=None, test_num=10):
    date_list = TRAIN_DATE_LIST if simulator.experiment_mode == 'train' else TEST_DATE_LIST
            
    for date in date_list:
        simulator.experiment_date = date
        for num in range(test_num):
            print('test round: ', num)
            simulator.reset()
            for _ in range(simulator.finish_run_step):
                simulator.step()
            
            logger.info(f"epoch: {num} == total_reward:{simulator.total_reward},matching_rate:{simulator.matched_requests_num/simulator.total_request_num},total_request_num:{simulator.total_request_num},matched_request_num:{simulator.matched_requests_num},occupancy_rate:{simulator.occupancy_rate},occupancy_rate_no_pickup:{simulator.occupancy_rate_no_pickup},wait_time:{simulator.waiting_time / simulator.matched_requests_num},pickup_time:{simulator.pickup_time / simulator.matched_requests_num}")
          
        
def simulation_greedy(simulator, test_num, logger):
    date_list = TRAIN_DATE_LIST if simulator.experiment_mode == 'train' else TEST_DATE_LIST
            
    for date in date_list:
        simulator.experiment_date = date
        for num in range(test_num):
            print('test round: ', num)
            simulator.reset()
            for _ in range(simulator.finish_run_step):
                # observe the transition and store the transition in the replay buffer (simulator.dispatch_transitions_buffer)
                simulator.step()
            logger.info(f"epoch: {num} == total_reward:{simulator.total_reward},matching_rate:{simulator.matched_requests_num/simulator.total_request_num},total_request_num:{simulator.total_request_num},matched_request_num:{simulator.matched_requests_num},occupancy_rate:{simulator.occupancy_rate},occupancy_rate_no_pickup:{simulator.occupancy_rate_no_pickup},wait_time:{simulator.waiting_time / simulator.matched_requests_num},pickup_time:{simulator.pickup_time / simulator.matched_requests_num}")
            

    test_num = 10
    for num in range(test_num):
        print('test round: ', num)

        date_list = None
        date_list = TRAIN_DATE_LIST if simulator.experiment_mode == 'train' else TEST_DATE_LIST
        
        for date in date_list:
            simulator.experiment_date = date
            simulator.reset()
            start_time = time.time()
            simulator.wait_requests['matching_radius'] = 0.5
            for step in range(simulator.finish_run_step):
                simulator.step()
            end_time = time.time()
            logger.info(f"epoch : {num} == total_reward:{simulator.total_reward},matching_rate:{simulator.matched_requests_num/simulator.total_request_num},total_request_num:{simulator.total_request_num},matched_request_num:{simulator.matched_requests_num},occupancy_rate:{simulator.occupancy_rate},occupancy_rate_no_pickup:{simulator.occupancy_rate_no_pickup},wait_time:{simulator.waiting_time / simulator.matched_requests_num},pickup_time:{simulator.pickup_time / simulator.matched_requests_num}")