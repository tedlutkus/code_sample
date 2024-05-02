"""
Code Sample - Theodore Lutkus
Here are a few methods I've implemented from modules I've been working on to:
1. Learn the system dynamics of physical systems (Feedforward MLP here)
2. Apply control techniques on those learned dynamics (RL, MPC, classical control on linearized MLP)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pickle
import gymnasium as gym
from stable_baselines3 import PPO
import torch
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import mlp_gym_pkg

# Train PPO policy on vectorized environment that uses MLP as simulator of system dynamics
def train_ppo_mlpsim_vec(num_states, num_actions, state_limits, action_limits,
                 model, model_scaler, model_dt, model_buffer_sizes, model_buffer_spacings,
                 rl_policy_history_length, learning_steps):
    # Create vectorized environment
    num_proc = 4
    env_list = []
    seed = 0
    for i in range(num_proc):        
        def _init():
            env = gym.make('mlp_gym_pkg/MlpSim',
                            num_states=num_states, 
                            num_actions=num_actions, 
                            state_limits=state_limits, 
                            action_limits=action_limits,
                            model=model, 
                            model_scaler=model_scaler, 
                            model_dt=model_dt, 
                            model_buffer_sizes=model_buffer_sizes, 
                            model_buffer_spacings=model_buffer_spacings,
                            rl_policy_history_length=rl_policy_history_length)
            env = Monitor(env)
            env.reset(seed=seed + i)
            return env
        set_random_seed(seed)
        env_list.append(_init)

    # Train PPO policy
    vec_env = DummyVecEnv(env_list)
    model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="./rl_policies/ppo_tensorboard/")
    model.learn(total_timesteps=learning_steps, progress_bar=True)

    return model

# MPC via Monte Carlo rollouts using MLP model and selecting lowest cost
class RandomShootingMPC:
    def __init__(self, Q, R, horizon, num_trajectory_samples, 
                 state_limits, action_limits, model, model_dt):
        # MPC parameters
        self.Q = Q
        self.R = R
        self.horizon = horizon
        self.num_trajectory_samples = num_trajectory_samples
        self.state_limits = state_limits
        self.action_limits = action_limits
        self.last_action = 0
        self.velocity = np.zeros((self.num_trajectory_samples, 1))
        self.position = np.zeros((self.num_trajectory_samples, 1))
        self.position_plot = np.zeros((self.num_trajectory_samples, self.horizon))

        # Neural network variables and buffers
        self.model = model
        self.model_dt = model_dt

    def reset_mpc(self):
        self.last_action = 0
       
    def mpc_step(self, current_state, goal_state, return_trajectories=False):
        # Generate brownian action trajectories
        noise = np.random.normal(loc=0.0, scale=0.25, size=(self.num_trajectory_samples, self.horizon))
        action_trajectories = np.cumsum(noise, axis=1)
        action_trajectories[0, :] = goal_state[0] # Add default controller as option for MPC
        action_trajectories = np.clip(action_trajectories, -self.action_limits[0], self.action_limits[0])
        
        # Simulate trajectories, calculate cost
        self.position[:, 0] = current_state[0]
        self.velocity[:, 0] = current_state[1]
        cost = np.zeros((self.num_trajectory_samples))
        model_inputs = np.zeros((self.num_trajectory_samples, 2))
        self.position_plot = np.zeros((self.num_trajectory_samples, self.horizon))
        for i in range(self.horizon):
            # Simulate next step of sampled trajectories
            model_inputs[:, 0] = self.velocity[:, 0]
            model_inputs[:, 1] = action_trajectories[:, i] - self.position[:, 0]
            model_input_tensor = torch.tensor(model_inputs, dtype=torch.float32)
            with torch.no_grad():
                state_delta_predictions = self.model(model_input_tensor).numpy()
            self.velocity[:, 0] += state_delta_predictions[:, 0]
            self.position[:, 0] += self.velocity[:, 0] * self.model_dt
            if return_trajectories:
                self.position_plot[:, i] = self.position[:, 0]
            
            # Accumulate next step cost, quadratic cost on state error with Q and on action_trajectories[:, i] with R
            state = np.hstack((self.position, self.velocity))
            state_error = state - goal_state
            state_cost = np.sum(state_error @ self.Q * state_error, axis=1)
            cost += state_cost

        action_change = abs(action_trajectories[:, 0].reshape(-1, 1) - self.last_action)
        cost += np.sum(action_change @ self.R * action_change, axis=1)

        # Select trajectory with lowest cost
        action_index = np.argmin(cost)
        best_action = action_trajectories[action_index, 0]
        self.last_action = best_action
        if return_trajectories:
            return best_action, self.position_plot, self.position_plot[action_index, :]
        else:
            return best_action

# Assumes data formatted as: [outputs, inputs, episode_terminated]
def normalize_data(data: pd.DataFrame, episodic=False):
    cols_to_normalize = data.columns
    if episodic:
        # Normalize all columns except the last (terminated)
        cols_to_normalize = data.columns[:-1]
    
    # Fit scaler to dataframe and transform dataframe
    scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data[cols_to_normalize].values)
    normalized_df = pd.DataFrame(normalized_data, columns=cols_to_normalize, index=data.index)
    
    if episodic:
        normalized_df = pd.concat([normalized_df, data.iloc[:, -1:]], axis=1)

    return normalized_df, scaler

def save_normalizing_scaler(scaler: MinMaxScaler, save_dir='mlp_models/scaler.pkl'):
    with open(save_dir, 'wb') as f:
        pickle.dump(scaler, f)
        
def load_normalizing_scaler(load_dir='mlp_models/scaler.pkl'):
    with open(load_dir, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

# Generate tensors for training network
# Assumes data formatted as: [outputs, inputs, episode_terminated]
def training_data(data: pd.DataFrame, num_outputs, num_inputs):
    y = data.iloc[:, :num_outputs].values
    X = data.iloc[:, num_outputs:num_outputs+num_inputs].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    return X_tensor, y_tensor

# Expects data to be formatted: [outputs, inputs, episode_terminated]
# Buffers always assume time increasing to the right
def training_data_with_buffers(data: pd.DataFrame, num_outputs, num_inputs, buffer_sizes: list, buffer_spacings: list):    
    # Determine episode boundaries
    episode_bounds = np.where(data.iloc[:, -1] == 1)[0]
    episode_bounds = np.append(0, episode_bounds)
    
    # Preallocate X and y arrays
    total_outputs = num_outputs + sum(buffer_sizes[:num_outputs])
    total_inputs = num_inputs + sum(buffer_sizes[num_outputs:num_outputs+num_inputs])
    num_samples = len(data)
    X = np.zeros((num_samples, total_inputs))
    y = np.zeros((num_samples, total_outputs))
    
    # Apply forward buffer to outputs
    # Formatting: [y, y_t+1, y_t+2, y1, y1_t+1, ...]
    idx = 0
    for i in range(num_outputs):
        buffer_size = buffer_sizes[i]
        buffer_spacing = buffer_spacings[i]
        for j in range(buffer_size+1):
            y[:, idx] = np.roll(data.iloc[:, i], -j*buffer_spacing)
            # Stop forward buffer from overlapping into each next episode
            for bound in episode_bounds[1:]:
                y[bound-(j*buffer_spacing):bound, idx] = 0
            idx += 1
    
    # Apply history buffer to inputs
    # Formatting: [x_t-2, x_t-1, x, x1_t-2, ...]
    for i in range(num_inputs):
        buffer_size = buffer_sizes[num_outputs+i]
        buffer_spacing = buffer_spacings[num_outputs+i]
        idx = i + sum(buffer_sizes[num_outputs:num_outputs+i+1]) # Formatting via indexing final X
        for j in range(buffer_size+1):
            X[:, idx] = np.roll(data.iloc[:, num_outputs+i], j*buffer_spacing)
            # Stop history buffer from overlapping with each previous episode
            for bound in episode_bounds[:-1]:
                X[bound:bound+(j*buffer_spacing), idx] = 0
            idx -= 1
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

# 1D history buffer formatted as eg: [t-4, t-2, t] - size=3, spacing=2
class HistoryBuffer:
    def __init__(self, buffer_size, buffer_spacing):
        # Buffer stored as numpy array to be referenced on its own as well
        self.buffer = np.zeros((1, (buffer_size*buffer_spacing)+1), dtype=np.float32)
        if buffer_spacing == 0:
            self.buffer_index = np.array([0])
        else:
            self.buffer_index = np.arange(-((buffer_size*buffer_spacing)+1), 0, buffer_spacing)

    # Returns formatted buffer of specified size and spacing between elements
    def get_buffer(self):
        return self.buffer[0, self.buffer_index]
    
    def update(self, value):
        self.buffer = np.roll(self.buffer, -1)
        self.buffer[-1] = value
        
    def reset(self):
        self.buffer.fill(0.)

# Group of history buffers for convenience 
class HistoryBufferGroup:
    def __init__(self, buffer_sizes, buffer_spacings):
        self.buffer_list = []
        for i in range(len(buffer_sizes)):
            self.buffer_list.append(HistoryBuffer(buffer_sizes[i], buffer_spacings[i]))

    def get_flattened_buffers(self):
        flattened_buffer = np.array([])
        for buffer in self.buffer_list:
            flattened_buffer = np.concatenate((flattened_buffer, buffer.get_buffer()), dtype=np.float32)
        return flattened_buffer
    
    def get_buffer(self, buffer_index):
        return self.buffer_list[buffer_index].get_buffer()
    
    def update(self, buffer_index, value):
        self.buffer_list[buffer_index].update(value)
    
    def update_all(self, values):
        for i, buffer in enumerate(self.buffer_list):
            buffer.update(values[i])
        
    def reset_all(self):
        for buffer in self.buffer_list:
            buffer.reset()

# General MLP class
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=2, hidden_size=32, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        layers = []
        
        # Add first hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(activation)
        
        # Add remaining hidden layers
        for _ in range(hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        
        # Add output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Train mlp model from prepared X and y tensors
def train_mlp_model(X_tensor, y_tensor, hidden_layers=2, hidden_size=32, 
                    activation_func=nn.ReLU(), num_epochs=100, learning_rate=0.001, plot_results=True):

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

    # Initialize the model, loss function, and optimizer
    model = MLP(input_size=X_train.shape[1], output_size=y_train.shape[1], hidden_layers=hidden_layers, hidden_size=hidden_size, activation=activation_func)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        # Testing
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Test Loss: {test_loss.item()}')
            
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Training and Testing Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Generate predictions using the trained model
        with torch.no_grad():
            model.eval()
            example_outputs = model(X_test[:200])

        # Plot actual vs predicted output
        plt.figure(figsize=(12, 6))
        plt.plot(y_test[:200], label='Actual state')
        plt.plot(example_outputs.numpy()[:200], '.', label='Predicted state')
        plt.title('Actual vs Predicted State')
        plt.xlabel('Sample')
        plt.ylabel('State Value')
        plt.legend()
        plt.show()
            
    return model

# Linearize the mlp dynamics around an operating point, option to modify function used by jacobian
def linearize_mlp(model, operating_point, mlp_output_func=None):
    if mlp_output_func is None:
        # Default to assumption that model output is exactly state_dot
        def default_mlp_output_func(input):
            return model(input)
        forward_func = default_mlp_output_func
    else:
        # Otherwise use a custom provided function
        forward_func = lambda input: mlp_output_func(input, model)
    
    jacobian = torch.autograd.functional.jacobian(forward_func, operating_point)
    
    num_states = jacobian.size(0)
    A = jacobian[:, :num_states].clone().detach().numpy()
    B = jacobian[:, num_states:].clone().detach().numpy()

    return A, B

# Linearize a timestep of mlp_based dynamics from inputs and outputs with tracked gradients
def linearize_timestep(state_dot, state):
    num_states = state_dot.size(0)
    jacobian_matrix = np.zeros((num_states, state.size(0)))

    for i in range(num_states):
        grad_input = torch.autograd.grad(state_dot[i], state, retain_graph=True)[0]
        jacobian_matrix[i] = grad_input.detach().numpy()
    
    A = jacobian_matrix[:, :num_states]
    B = jacobian_matrix[:, num_states:]
    
    return A, B