from collections import deque
import phyre
import random
import numpy as np
from Model import SequencePrediction
from Train import train_model
import utils
import torch
import torch.optim as optim
from sklearn.utils import shuffle
from datetime import datetime

# TODO: Randomize the dataset
# TODO: No invalid state

# COPIED PHYRE VARIABLE FOR REFERENCE

"""
DEFAULT_MAX_STEPS = 1000
FPS = 60
OBJECT_FEATURE_SIZE = 14
STEPS_FOR_SOLUTION = 180
"""

if torch.cuda.is_available():
    vdevice = torch.device("cuda")
else:
    vdevice = torch.device("cpu")

start_time = datetime.now()

sequence_model = SequencePrediction(vdevice)
sequence_model.to(vdevice)
model_optimizer = optim.Adam(sequence_model.parameters(), lr=utils.LEARNING_RATE)
loss_fn = torch.nn.MSELoss(reduction='sum')

tier = 'ball'
eval_setup = 'ball_cross_template'
fold_id = 0 # For simplicity, we will just use one fold for evaluation.

train, dev, test = phyre.get_fold(eval_setup, fold_id)
# Smaller Dataset
train = shuffle(train, random_state=42)[:10]

cache = phyre.get_default_100k_cache(tier)
dataset = []
simulator = phyre.initialize_simulator(train, tier)
evaluator = phyre.Evaluator(train)
total_size = len(train)
batch_size = 0
IMAGE_SHAPE = utils.IMAGE_SHAPE

training_counter = 0
sequence_network_loss, reward_network_loss = deque(maxlen=500), deque(maxlen=500)
sequence_network_loss_deque = deque(maxlen=5)
reward_network_loss_deque = deque(maxlen=5)
utils.clean_output_dir()

for epoch in range(utils.MAX_EPOCH):
    random.seed()
    index_list = list(range(total_size))    # 2 random puzzles
    random.shuffle(index_list)
    for iter, task_index in enumerate(index_list):
        task_id = train[task_index]
        batch_size += 1
        if iter % 20 == 0:
            utils.log(str(total_size - iter) + " left to process for epoch : " + str(epoch))
            print(str(total_size - iter) + " left to process for epoch : " + str(epoch))
        statuses = cache.load_simulation_states(task_id)
        cached_status = phyre.simulation_cache.INVALID  # 0
        # For each task take 9 random actions and 1 correct action
        for _ in range(9):
            action, cached_status, _ = utils.get_random_valid_action(cache, statuses)
            simulation = simulator.simulate_action(task_index, action, need_images=True)
            initial_scene = phyre.vis.observations_to_float_rgb(simulator.initial_scenes[task_index])
            initial_scene = utils.process_image(initial_scene, target_shape=IMAGE_SHAPE)
            imgs = np.array(list(map(lambda img: phyre.vis.observations_to_float_rgb(img), simulation.images)))
            while imgs.shape[0] < utils.HORIZON:  # appending the last frame to the end
                imgs = np.vstack((imgs, imgs[-1::]))
            imgs = utils.process_images(imgs, target_shape=IMAGE_SHAPE)
            dataset.append(utils.construct_dataset_entry(initial_scene=initial_scene, images=imgs, action=action,
                                                         task_id=task_id, action_status=cached_status))

        # Now let's create a simulator for this task to simulate the action.
        solving_action_index, solving_status = utils.get_solving_action(statuses, cache)
        solving_action = cache.action_array[solving_action_index]
        simulation = simulator.simulate_action(task_index, solving_action, need_images=True)
        initial_scene = phyre.vis.observations_to_float_rgb(simulator.initial_scenes[task_index])
        initial_scene = utils.process_image(initial_scene, target_shape=IMAGE_SHAPE)
        imgs = np.array(list(map(lambda img: phyre.vis.observations_to_float_rgb(img), simulation.images)))
        imgs = utils.process_images(imgs, target_shape=IMAGE_SHAPE)

        while imgs.shape[0] < utils.HORIZON: # appending the last frame to the end
            imgs = np.vstack((imgs, imgs[-1::]))
        dataset.append(utils.construct_dataset_entry(initial_scene=initial_scene, images=imgs, action=solving_action,
                                               task_id=task_id, action_status=solving_status))

        if batch_size >= utils.TOTAL_BATCH_SIZE or iter == len(train):
            training_counter+=1
            batch_size = 0
            x, target, target_reward, action = utils.create_input(dataset)
            x, target, target_reward, action = shuffle(x, target, target_reward, action)    # NOTE: Should be consistent but double-check
            dataset = []
            image_shape = target.shape[2]
            seq_loss, reward_loss, prediction = train_model(sequence_model, model_optimizer, vdevice, loss_fn,
                                                            x, target, target_reward, action, image_shape)
            utils.log(f'(Seq_Loss, Reward_loss): ({seq_loss}, {reward_loss})')
            sequence_network_loss_deque.append(seq_loss)
            reward_network_loss_deque.append(reward_loss)

            if training_counter % 5 == 0 and training_counter > 0:
                sequence_network_loss.append(np.mean(sequence_network_loss_deque))
                reward_network_loss.append(np.mean(reward_network_loss_deque))
                print(f'(Seq_Loss, Reward_loss): ({np.mean(sequence_network_loss_deque)}, {np.mean(reward_network_loss_deque)}) Training_Counter = {training_counter}')
                utils.log(f'(Seq_Loss, Reward_loss): ({np.mean(sequence_network_loss_deque)}, {np.mean(reward_network_loss_deque)}) Training_Counter = {training_counter}')

    # if epoch > 5:
    #    utils.save_imgs(prediction[0], target[0], epoch, utils.IMAGE_SHAPE)
end_time = datetime.now()
total_time = end_time - start_time

print(f"TOTAL EXECUTION TIME : {total_time}")
utils.plot_loss(list(sequence_network_loss), list(reward_network_loss))

utils.save_imgs(prediction[0], target[0], epoch, utils.IMAGE_SHAPE, "batch0")
utils.save_imgs(prediction[1], target[1], epoch, utils.IMAGE_SHAPE, "batch1")
utils.save_imgs(prediction[2], target[2], epoch, utils.IMAGE_SHAPE, "batch2")
utils.save_imgs(prediction[3], target[3], epoch, utils.IMAGE_SHAPE, "batch3")