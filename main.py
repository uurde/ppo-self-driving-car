# PPO Implementation
 
# Installing the libraries
# !pip install --force-reinstall box2d-py==2.3.10
# !pip install gymnasium[box2d]
# !pip install torch gymnasium stable-baselines3 gymnasium[box2d] torchvision
 
# Importing the libraries
import ale_py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from collections import deque
 
# Setting up the hyperparameters
gamma = 0.99
lambda_gae = 0.95
lr = 3e-4
clip_epsilon = 0.2
entropy_coef = 0.01
value_loss_coef = 0.5
ppo_epochs = 10
batch_size = 64
num_steps = 2048
 
# Setting up the CarRacing Environment
env = gym.make("CarRacing-v3")
 
# Building the Neural Network for Actor-Critic
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # CNN to handle image input
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Fully connected layers
        dummy_input = torch.zeros(1, 3, 96, 96)
        dummy_output = self.conv(dummy_input)
        flattened_size = dummy_output.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            #nn.Linear(64 * 9 * 9, 512),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
        )
        # Actor (Policy) and Critic (Value function) heads
        self.actor_mean = nn.Linear(512, 3)  # 3 outputs (steering, gas, brake)
        self.actor_log_std = nn.Parameter(torch.zeros(1, 3))
        self.critic = nn.Linear(512, 1)
    def forward(self, x):
        x = x / 255.0  # Normalize pixel values
        x = self.conv(x)
        flattened_size = x.size(1) * x.size(2) * x.size(3)
        x = x.reshape(x.size(0), flattened_size)
        #x = x.view(x.size(0), flattened_size)
        #x = x.view(x.size(0), -1)
        x = self.fc(x)
        mean = torch.tanh(self.actor_mean(x))
        std = torch.exp(self.actor_log_std)
        value = self.critic(x)
        return mean, std, value
 
# Building the PPO Agent
class PPOAgent:
    def __init__(self):
        self.model = ActorCritic()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            mean, std, _ = self.model(state)
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(-1.0, 1.0)  # Clip actions within valid range
        return action.numpy()[0], dist.log_prob(action).sum().item(), dist.entropy().sum().item()
    def compute_returns(self, rewards, masks, values, next_value):
        returns = []
        gae = 0
        values = values + [next_value]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lambda_gae * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    def update(self, trajectories):
        states, actions, log_probs, returns, values, advantages = trajectories
        states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2)
        actions = torch.tensor(actions, dtype=torch.float32)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        for _ in range(ppo_epochs):
            mean, std, new_values = self.model(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()
            # Compute ratio (new / old probabilities)
            ratio = (new_log_probs - log_probs).exp()
            # Surrogate loss (PPO)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            # Critic loss (value function)
            value_loss = (returns - new_values).pow(2).mean()
            # Total loss
            loss = actor_loss + value_loss_coef * value_loss - entropy_coef * entropy
            # Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
 
# Implementing the training loop
def train(agent, env, num_episodes=100):
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False
        score = 0
        # Storage for experience
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        masks = []
        entropies = []
        for step in range(num_steps):
            #Convert state to PyTorch tensor before passing to the model
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            action, log_prob, entropy = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            mask = 1 - done
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            masks.append(mask)
            entropies.append(entropy)
            #values.append(agent.model(state)[2].item())
            values.append(agent.model(state_tensor)[2].item())
            state = next_state
            score += reward
            if done or truncated:
                break
        # Compute returns
        next_state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        next_value = agent.model(next_state_tensor)[2].item()
        returns = agent.compute_returns(rewards, masks, values, next_value)
        advantages = torch.tensor(returns) - torch.tensor(values)
        # Update agent
        agent.update((states, actions, log_probs, returns, values, advantages))
        print(f'Episode: {episode}, Score: {score}')
 
# Creating the agent and training it on the CarRacing Environment
agent = PPOAgent()
train(agent, env)
 
# Visualizing the final results
import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder
def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)
show_video_of_model(agent, 'CarRacing-v3')
# def show_video():
#     mp4list = glob.glob('*.mp4')
#     if len(mp4list) > 0:
#         mp4 = mp4list[0]
#         video = io.open(mp4, 'r+b').read()
#         encoded = base64.b64encode(video)
#         display(HTML(data='''<video alt="test" autoplay
#                 loop controls style="height: 400px;">
#                 <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#              </video>'''.format(encoded.decode('ascii'))))
#     else:
#         print("Could not find video")
# show_video()