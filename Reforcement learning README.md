# Reinforcement Learning for Trading

## Project Overview
This project leverages Reinforcement Learning (RL) to develop trading strategies using TensorFlow and Gym. The notebook showcases the implementation of a custom trading environment, designed to simulate the trading process while enabling agents to learn optimal policies through interaction with market data.

## Key Concepts Applied

- **Reinforcement Learning Fundamentals**: Utilized core concepts of reinforcement learning, including agents, environments, states, actions, and rewards, tailored specifically for trading applications.
- **Custom Trading Environment**: Developed a custom environment based on OpenAI's Gym to simulate trading scenarios and enable the training of RL agents.
- **Agent Implementation**: Implemented a Double Dueling DQN agent to learn and make trading decisions based on market data.
- **Data Handling and Processing**: Processed financial data to serve as input for the RL model, applying techniques such as downsampling and normalization.

## Getting Started

### Prerequisites

To run this notebook, ensure you have the following:

- Python version ≥ 3.7
- TensorFlow version ≥ 2.8
- Required libraries:
  - NumPy
  - Pandas
  - Matplotlib
  - OpenAI Gym
  - Stable Baselines3

You can install the necessary libraries using pip:

```bash
pip install numpy pandas matplotlib gym stable-baselines3
```

### Running the Notebook

To execute the notebook, follow these steps:

1. Clone or download the repository containing the notebook.
2. Open the notebook in a Jupyter environment.
3. Run the cells sequentially to set up the trading environment, implement the RL agent, and visualize results.

## Key Concepts Covered

### 1. Setup

The notebook begins by importing necessary libraries and ensuring that the environment is correctly configured. Here’s a sample of the initial setup:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import tensorflow as tf
```

### 2. Custom Trading Environment

A custom trading environment is defined using OpenAI Gym, allowing the reinforcement learning agent to interact with market data. The environment includes action spaces for buying, selling, or holding assets:

```python
class Actions(Enum):
    Sell = 0
    Buy = 1
    Rest = 2

class CustomTradingEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size):
        self.df = df
        self.window_size = window_size
        self.action_space = spaces.Discrete(len(Actions))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, df.shape[1]), dtype=np.float32)
```

### 3. Data Processing

The notebook includes functionality to load and preprocess financial data, preparing it for the trading environment:

```python
data = pd.read_csv('USDCHF_Candlestick_4_Hour_BID_14.12.2007-14.12.2022.csv')
```

### 4. Reinforcement Learning Agent

A Double Dueling DQN agent is implemented to learn from the trading environment. This agent makes decisions based on the current state of the market:

```python
from stable_baselines3 import DQN

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
```

### 5. Reward Calculation

The reward structure is critical for the agent's learning. The notebook implements a function to calculate rewards based on the actions taken:

```python
def _calculate_reward(self, action):
    step_reward = 0
    if action == Actions.Buy.value:
        step_reward += self.price_diff * 10000  # Example reward calculation
    return step_reward
```

### 6. Visualization and Results

The notebook includes visualization of the trading performance, allowing for assessment of the agent's decision-making over time:

```python
plt.plot(self.df['Close'])
plt.title('Trading Performance')
plt.show()
```

## Conclusion

This project serves as a comprehensive resource for applying reinforcement learning to trading scenarios. By developing a custom trading environment and implementing a learning agent, users can gain insights into the complexities of financial markets and the potential of RL strategies.
