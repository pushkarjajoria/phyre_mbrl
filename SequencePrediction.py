import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from RewardPrediction import RewardPrediction


PHYRE_MAX_TIMESTEPS = 17
FEATURES_PER_TIMESTEP = 10   # Small 15x15 image to deconv
DECONV_SHAPE = 1
DECONV_CHANNELS = 1024
CHANNELS = 1
IMAGE_SIZE = 64

INPUT_SIZE = 1024
HIDDEN_SIZE = 1024
NUM_LAYERS = 1


class SequencePrediction(nn.Module):

    def __init__(self, vdevice):
        super(SequencePrediction, self).__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 32, (4, 4), (2, 2))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), (2, 2))
        self.conv3 = nn.Conv2d(64, 128, (4, 4), (2, 2))
        self.conv4 = nn.Conv2d(128, 256, (4, 4), (2, 2))
        self.drop = nn.Dropout(.2)
        self.fc1 = nn.Linear(1024, 482)

        # Sequence Prediction
        self.fc2 = nn.Linear(512, 1024)
        self.rnn = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc_sequence_decoder = nn.Linear(1024, 1024)
        self.deconv1 = nn.ConvTranspose2d(DECONV_CHANNELS, 128, (5, 5), stride=(2, 2))
        self.deconv2 = nn.ConvTranspose2d(128, 64, (5, 5), stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(64, 32, (6, 6), stride=(2, 2))
        self.deconv4 = nn.ConvTranspose2d(32, CHANNELS, (6, 6), stride=(2, 2))

        # Reward Prediction
        self.fc5 = nn.Linear(512, 1)
        self.vdevice = vdevice

    def forward(self, x, action):
        batch_size = x.shape[0]
        # Convolution
        conv1 = F.tanh(self.conv1(x.float()))
        conv2 = F.tanh(self.conv2(conv1))
        conv3 = F.tanh(self.conv3(conv2))
        conv4 = F.tanh(self.conv4(conv3))
        v_to = 1
        for i in range(1, len(conv4.shape)):
          v_to = v_to*conv4.shape[i]
        flatten_output = conv4.view(batch_size, v_to)
        fc1 = self.fc1(flatten_output)
        z = torch.cat((fc1, action), dim=1)

        # sequence prediction
        fc2_out = F.leaky_relu(self.fc2(z.float()))
        hidden = (torch.randn(1, batch_size, 1024).to(self.vdevice),
                  torch.randn(1, batch_size, 1024).to(self.vdevice))
        rnn_output = []
        rnn_inp = fc2_out.view(50, 1, 1024)
        for i in range(17):
            rnn_inp, hidden = self.rnn(rnn_inp, hidden)
            rnn_output.append(rnn_inp)
        rnn_output = torch.stack(rnn_output)
        h = F.leaky_relu(self.fc_sequence_decoder(rnn_output))
        h = h.view(batch_size * PHYRE_MAX_TIMESTEPS, DECONV_CHANNELS, DECONV_SHAPE, DECONV_SHAPE)
        deconv1_out = F.tanh(self.deconv1(h))
        deconv2_out = F.tanh(self.deconv2(deconv1_out))
        deconv3_out = F.tanh(self.deconv3(deconv2_out))
        obs_prediction = self.deconv4(deconv3_out)
        obs_prediction = obs_prediction.view(batch_size, PHYRE_MAX_TIMESTEPS, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)

        # reward prediction
        reward = self.fc5(z.float())

        return obs_prediction, reward