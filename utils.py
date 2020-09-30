from datetime import datetime
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import torch
from skimage import io
import random
import phyre
import cv2

CURRENT_FOLDER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
OUTPUT_FOLDER = 'output/'
OUTPUT_DIR = OUTPUT_FOLDER + CURRENT_FOLDER + '/'
IMAGE_SHAPE = 64

def display_img(img, grayscale=True):
    if grayscale:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.show()
    plt.waitforbuttonpress()
    plt.close()


def save_imgs(prediction, target, epoch, img_shape, save_id):
    if torch.cuda.is_available():
        prediction = prediction.cpu().detach().numpy()
    else:
        prediction = prediction.detach().numpy()

    if not os.path.exists(OUTPUT_DIR + "epoch"+str(epoch)):
        os.mkdir(OUTPUT_DIR + "epoch"+str(epoch))
    target_dir = OUTPUT_DIR + "epoch"+str(epoch) + '/target'
    prediction_dir = OUTPUT_DIR + "epoch"+str(epoch) + '/prediction'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    if not os.path.exists(prediction_dir):
        os.mkdir(prediction_dir)

    for i,img in enumerate(target):
        img = np.reshape(img, (img_shape, img_shape))
        plt.imsave(target_dir + '/img_' + str(i) + "_" + save_id + '.jpg', img)

    for i,img in enumerate(prediction):
        img = np.reshape(img, (img_shape, img_shape))
        plt.imsave(prediction_dir + '/img_' + str(i) + "_" + save_id + '.jpg', img)


def plot_loss(seq_loss, reward_loss):
    plt.plot(seq_loss)
    plt.savefig(OUTPUT_DIR + "sequence_loss.png")
    plt.close()

    plt.plot(reward_loss)
    plt.savefig(OUTPUT_DIR + "reward_loss.png")
    plt.close()


def clean_output_dir():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    # if os.path.exists(OUTPUT_DIR + "log.txt"):
    #     os.remove(OUTPUT_DIR + "log.txt")
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    os.mkdir(OUTPUT_DIR)
    f = open(OUTPUT_DIR + "log.txt", "w+")
    f.close()


def log(txt):
    f = open(OUTPUT_DIR + "log.txt", "a+")
    timestamp_str = datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)")
    f.write(timestamp_str + " : " + txt + "\n")
    f.close()


def process_image(img, target_shape=IMAGE_SHAPE, gray_scale=True):
    if gray_scale:
        img = np.float32(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img / np.max(img)     # Normalization
    return cv2.resize(img, dsize=(target_shape, target_shape), interpolation=cv2.INTER_CUBIC)


def process_images(imgs, target_shape=IMAGE_SHAPE):
    resized_images = []
    for i, img in enumerate(imgs):
        resized_images.append(process_image(img, target_shape=target_shape))
    return np.array(resized_images)


def get_random_valid_action(cache, statuses):
    cached_sts = 0
    while cached_sts == 0:
        action_index = random.randint(0, len(cache)-1)
        acn = cache.action_array[action_index]
        cached_sts = statuses[action_index]

    return acn, cached_sts, action_index


def get_solving_action(statuses, cache):
    for i, v in enumerate(statuses):
        if v == phyre.SimulationStatus.SOLVED:
            return i, phyre.SimulationStatus.SOLVED
    # If no solving actions, return a random valid action
    _, status, acn_index = get_random_valid_action(cache)
    return acn_index, status


def construct_dataset_entry(initial_scene, images, action, task_id, action_status):
    return {"input_image": np.array(initial_scene),
            "target_images": np.array(images),
            "action": np.array(action),
            "task_id": task_id,
            "action_status": action_status}


def create_input(dataset):
    x = []  # Input scene
    y = []  # Y sequence
    y_reward = []   # Y Reward
    a = []  # Action
    for entry in dataset:
        x.append(entry['input_image'])
        y.append(entry['target_images'])
        y_reward.append((entry['action_status']))
        a.append(entry['action'])

    return np.array(x), np.array(y), np.array(y_reward), np.array(a)


