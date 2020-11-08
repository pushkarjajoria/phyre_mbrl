from typing import Any

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import utils


class SequencePrediction(nn.Module):

    def __init__(self, vdevice):
        super(SequencePrediction, self).__init__()
        self.conv1 = nn.Conv2d(utils.CHANNELS, 32, (4, 4), (2, 2))
        self.conv2 = nn.Conv2d(32, 64, (4, 4), (2, 2))
        self.conv3 = nn.Conv2d(64, 128, (4, 4), (2, 2))
        self.conv4 = nn.Conv2d(128, 256, (4, 4), (2, 2))
        self.drop = nn.Dropout(.2)
        self.fc1 = nn.Linear(1024, 1009)

        # Sequence Prediction
        self.fc2 = nn.Linear(512, 1024)
        self.rnn = nn.LSTM(utils.RNN_INPUT_SIZE, utils.HIDDEN_SIZE, utils.NUM_LAYERS, batch_first=True)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 2048)
        self.fc_sequence_decoder = nn.Linear(1024, 1024)
        self.deconv1 = nn.ConvTranspose2d(utils.DECONV_CHANNELS, 128, (5, 5), stride=(2, 2))
        self.deconv2 = nn.ConvTranspose2d(128, 64, (5, 5), stride=(2, 2))
        self.deconv3 = nn.ConvTranspose2d(64, 32, (6, 6), stride=(2, 2))
        self.deconv4 = nn.ConvTranspose2d(32, utils.CHANNELS, (6, 6), stride=(2, 2))

        # Reward Prediction
        self.fc5 = nn.Linear(1024, 1)
        self.vdevice = vdevice

    def forward(self, x, action):
        batch_size = x.shape[0]
        # Convolution
        conv1 = F.leaky_relu(self.conv1(x.float()))
        conv2 = F.leaky_relu(self.conv2(conv1))
        conv3 = F.leaky_relu(self.conv3(conv2))
        conv4 = F.leaky_relu(self.conv4(conv3))
        v_to = 1
        for i in range(1, len(conv4.shape)):
          v_to = v_to*conv4.shape[i]
        flatten_output = conv4.view(batch_size, v_to)
        fc1 = self.fc1(flatten_output)
        z = torch.cat((fc1, action), dim=1)

        # sequence prediction
        hidden = (torch.randn(1, batch_size, 1024).to(self.vdevice),
                  torch.randn(1, batch_size, 1024).to(self.vdevice))
        rnn_output = []
        rnn_inp = z.view(batch_size, 1, utils.RNN_INPUT_SIZE)
        for i in range(utils.HORIZON):
            rnn_inp, hidden = self.rnn(rnn_inp, hidden)
            rnn_output.append(rnn_inp)
        rnn_output = torch.stack(rnn_output)
        rnn_output = rnn_output.permute(1, 0, 2, 3)
        h = F.leaky_relu(self.fc_sequence_decoder(rnn_output))
        h = h.permute(0, 1, 3, 2)
        h = h.view(batch_size * utils.HORIZON, utils.DECONV_CHANNELS, utils.DECONV_SHAPE, utils.DECONV_SHAPE)
        deconv1_out = F.leaky_relu(self.deconv1(h))
        deconv2_out = F.leaky_relu(self.deconv2(deconv1_out))
        deconv3_out = F.leaky_relu(self.deconv3(deconv2_out))
        obs_prediction = self.deconv4(deconv3_out)
        obs_prediction = obs_prediction.view(batch_size, utils.HORIZON, utils.CHANNELS, utils.IMAGE_SHAPE, utils.IMAGE_SHAPE)

        # reward prediction
        reward = self.fc5(z.float())

        return obs_prediction, reward