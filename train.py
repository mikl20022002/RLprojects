from dataclasses import dataclass
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import time


@dataclass
class Args:
    #training args
    seed: int = 0
    device: bool = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 32
    env_id: str = 'ALE/BattleZone-v5'
    steps: int = 200000
    training_freq: int = 4
    network_update_freq: int = 1000
    learning_start_step: int = 10000
    learning_rate: int = 5e-4
    discount: float = 0.95
    tau: float = 1e-3

    #epsilon args
    epsilon_decrease_start: int = 0
    epsilon_decrease_end: int = steps // 2
    epsilon_decrease_from: int = 1
    epsilon_decrease_to: int = 0
    epsilon_decrease_at_step: float = (epsilon_decrease_from - epsilon_decrease_to) / (epsilon_decrease_end - epsilon_decrease_start)

    #replay memory args
    buffer_size: int = steps // 5

args = Args()

hparams = {
    'batch_size' :args.batch_size,
    'steps' : args.steps,
    'training_freq' : args.training_freq,
    'network_update_freq' : args.network_update_freq,
    'learning_start_step' : args.learning_start_step,
    'learning_rate' : args.learning_rate,
    'buffer_size' : args.buffer_size,
    'discount' : args.discount,
    'tau' : args.tau,
    'epsilon_decrease_end' : args.epsilon_decrease_end
}

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.roi1_conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.roi1_conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.roi1_conv3 = nn.Conv2d(64, 64, 4, stride=2)

        self.roi2_conv1 = nn.Conv2d(4, 16, 4, stride=2)
        self.roi2_conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.linear1 = nn.Linear(1792, 512)
        self.linear2 = nn.Linear(512, 18)

    def forward(self, x):
        x = x / 255
        x1, x2 = self.ROI(x)

        # process ROI1
        x1 = F.relu(self.roi1_conv1(x1))
        # print('shape after roi1_conv1: ', x1.shape)
        x1 = F.relu(self.roi1_conv2(x1))
        # print('shape after roi1_conv2: ', x1.shape)
        x1 = F.relu(self.roi1_conv3(x1))
        # print('shape after roi1_conv3: ', x1.shape)
        x1 = x1.view(x1.shape[0], -1)

        #process ROI2
        x2 = F.relu(self.roi2_conv1(x2))
        # print('shape after roi2_conv1: ', x2.shape)
        x2 = F.relu(self.roi2_conv2(x2))
        # print('shape after roi2_conv2: ', x2.shape)
        x2 = x2.view(x2.shape[0], -1)

        x3 = torch.cat((x1, x2), dim=1)
        # print('shape after cat: ', x3.shape)
        x3 = self.linear1(x3)
        x3 = self.linear2(x3)
        return x3

    def ROI(self, img):
        """
        :return: roi1 - contains horizontal zone with tanks
                 roi2 - contains radar
        """
        height, width = img.shape[2], img.shape[3]

        roi1 = img[:, :, int(height * 0.4) : int(height * 0.75),:]
        roi2 = img[:, :, int(height * 0.02) : int(height * 0.17), int(width * 0.465) : int(width * 0.6) ]

        return roi1, roi2


def make_env(env_id, seed):
    env = gym.make(env_id, render_mode='rgb_array')
    # if "FIRE" in env.unwrapped.get_action_meanings():
        # env = gym.wrappers.FireResetEnv(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.ResizeObservation(env, (140, 110))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    env.action_space.seed(seed)
    return env

writer = SummaryWriter('runs/BattleZone_experement_1')

# Random seed definition
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Env init
env = make_env(args.env_id, args.seed)

# Replay memory
rm = ReplayBuffer(
    buffer_size = args.buffer_size,
    observation_space = env.observation_space,
    action_space = env.action_space,
    device = args.device,
    optimize_memory_usage = True,
    handle_timeout_termination=False
)

# Network init
q_network = QNetwork(env).to(args.device)
optimizer = torch.optim.Adam(q_network.parameters(), lr=args.learning_rate)
target_network = QNetwork(env).to(args.device)
target_network.load_state_dict(q_network.state_dict())

# Define epsilon
epsilon = args.epsilon_decrease_from

# Reset env
obs, _ = env.reset(seed=args.seed)

# Start process
for step in range(args.steps):
    if step % 5000 == 0: print(step)

    start_env_enteraction = time.time()

  # ЗАМЕНИЛ RANDOM.RANDINT(0, 1) НА RANDOM.UNIFORM(0, 1)
    if random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        q_values = q_network(torch.Tensor(obs).to(args.device).unsqueeze(0))
        action = torch.argmax(q_values, dim=1).cpu().numpy()
    # print(action.item())
    next_obs, reward, terminated, truncated, infos = env.step(action.item())
    # add correction for final obs
    rm.add(obs, next_obs, action, reward, terminated, infos)

    obs = next_obs

    # Update epsilon
    #ТЕПЕРЬ НЕ ВЫХОДИТ В ОТРИЦАТЕЛЬНУЮ СТОРОНУ
    epsilon = max(epsilon - args.epsilon_decrease_at_step, args.epsilon_decrease_to)

    if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], step)

    writer.add_scalar('env_cycle_time', time.time() - start_env_enteraction, step)

    if step > args.learning_start_step and step % args.training_freq == 0:

      start_training = time.time()

      data = rm.sample(args.batch_size)
      with torch.no_grad():
          target_max = target_network(data.next_observations).max(dim=1).values
          td_target = data.rewards.flatten() + args.discount * target_max * (1 - data.dones.flatten())
      old_val = q_network(data.observations).gather(1, data.actions).squeeze()
      loss = F.mse_loss(td_target, old_val)
      # print(f'step: {step}', f'loss: {loss.item()}')
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      writer.add_scalar('training_cycle_time', time.time() - start_training, step)

    if step > args.learning_start_step and step % args.network_update_freq == 0:

      start_params_transfer = time.time()

      for target_net_param, q_net_param in zip(target_network.parameters(), q_network.parameters()):
          target_net_param.data.copy_(
              q_net_param * args.tau + target_net_param * (1 - args.tau)
          )
      writer.add_scalar('param_transfer_time', time.time() - start_params_transfer, step)

metric = {'empty' : 0}
writer.add_hparams(hparam_dict=hparams, metric_dict=metric)
env.close()

#TODO: установить jupyter notebook