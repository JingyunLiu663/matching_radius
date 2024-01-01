# Deep Reinforcement Learning for Matching Radius Assignment for Order Dispatch 

- how to train the RL agent to adjust the radius dynamically
  ```
  python main.py -experiment_mode train -radius_method rl -rl_agent ddqn -cruise 1
  ```

  - experiment_mode: train/test
  - radius_method: rl for reinforcement learning to choose the matching radius dynamically
  - rl_agent: dqn/ddqn/dueling_dqn [comes with `rl` radius_method]
  - cruise: 1 - set the cruise flag to be True
 
- how to get the matching result for fixed radius
  ```
  python main.py -experiment_mode train -radius_method fixed -radius 1.0 -cruise 1 
  ```
  - experiment_mode: train/test
  - radius_method: fixed radius for the whole simulation
  - radius: set the fixed radius to a float number [comes with `fixed` radius method]
  - cruise: 1 - set the cruise flag to be True
