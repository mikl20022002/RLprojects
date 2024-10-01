import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from dataclasses import dataclass
import numpy as np
import random
import gymnasium as gym
from gymnasium.wrappers import GrayScaleObservation, FrameStack
import os
import time



class Policy(nn.Module):
    def __init__(self, env):
        super().__init__()

        self.roi1_conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.roi1_conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.roi1_conv3 = nn.Conv2d(64, 64, 4, stride=2)

        self.roi2_conv1 = nn.Conv2d(4, 16, 4, stride=2)
        self.roi2_conv2 = nn.Conv2d(16, 32, 4, stride=2)

        self.linear1 = nn.Linear(1792, 512)
        self.linear2 = nn.Linear(512, 18) #env.action.space

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
        # print('shape before softmax: ', x3.shape)
        x3 = F.softmax(x3, dim=1)
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
    env = gym.make(env_id, render_mode = 'human')
    env = GrayScaleObservation(env)
    env = FrameStack(env, 4)
    env.action_space.seed(seed)
    return env


def render(env, policy):
    obs, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            probs = policy(torch.tensor(np.asarray(obs)).unsqueeze(0))
            cat = Categorical(probs)
            action = cat.sample()
        obs, _, done, _, _ = env.step(action.item())
        env.render()


if __name__ == '__main__':
    env = make_env('ALE/BattleZone-v5', 0)
    policy = Policy(env)

    # path = 'models/model_at_step_3100_lr_1e-5.pth'
    # path = 'models/model_at_step_3100_lr_5e-5.pth'
    path = 'models/model_at_step_3100_pseudo_lr_1e-4.pth'
    parameters = torch.load(path, map_location=torch.device('cpu'))
    policy.load_state_dict(parameters)
    render(env, policy)