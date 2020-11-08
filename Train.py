import phyre
import torch
import torch.optim as optim
import numpy as np


def train_model(model, optimizer, vdevice, loss_fn, image, target, target_reward, action, image_shape, channels=1):
    batch_size = image.shape[0]
    image = np.reshape(image, (batch_size, image_shape, image_shape, channels))
    image = torch.from_numpy(image)
    image = image.permute(0, 3, 1, 2).to(vdevice)
    target = np.reshape(target, (batch_size, 17, channels, image_shape, image_shape))
    target = torch.from_numpy(target).to(vdevice)
    target_reward = torch.from_numpy(np.reshape(target_reward, (batch_size, 1))).to(vdevice)
    action = torch.from_numpy(np.repeat(action, 5, axis=1)).to(vdevice)     # Up sampling x5

    optimizer.zero_grad()
    seq_out, reward_out = model(image, action)

    loss_seq = loss_fn(seq_out.float(), target.float())
    normalized_loss_seq = loss_seq/(batch_size*image_shape*image_shape*channels)
    loss_reward = loss_fn(reward_out.float(), target_reward.float())
    normalized_loss_reward = loss_reward/batch_size
    normalized_total_loss = normalized_loss_seq + normalized_loss_reward
    normalized_total_loss.backward()
    optimizer.step()
    return float(normalized_loss_seq), float(normalized_loss_reward), seq_out
