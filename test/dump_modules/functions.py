import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import cv2
from scipy import ndimage
import math
import pickle


def correctness(TP, FP, eps=1e-12):
    return TP/(TP + FP + eps)  # precision


def completeness(TP, FN, eps=1e-12):
    return TP/(TP + FN + eps)  # recall


def quality(corr, comp, eps=1e-12):
    return (comp*corr)/(comp-comp*corr+corr+eps)


def relaxed_confusion_matrix(pred_s, gt_s, slack):

    distances_gt = ndimage.distance_transform_edt((np.logical_not(gt_s)))
    distances_pred = ndimage.distance_transform_edt((np.logical_not(pred_s)))

    true_pos_area_gt = distances_gt <= slack
    false_pos_area = distances_gt > slack

    true_pos_area_pred = distances_pred <= slack
    false_neg_area = distances_pred > slack

    # length of the matched extraction
    true_positives_gt = np.logical_and(true_pos_area_gt, pred_s).sum()
    false_positives = np.logical_and(false_pos_area, pred_s).sum()

    true_positives_pred = np.logical_and(
        true_pos_area_pred, gt_s).sum()  # length of the matched reference
    false_negatives = np.logical_and(false_neg_area, gt_s).sum()

    return true_positives_gt, true_positives_pred, false_negatives, false_positives


def compute_scores(TP_g, TP_p, FN, FP, eps=1e-12):
    corr = correctness(TP_g, FP, eps)
    comp = completeness(TP_p, FN, eps)
    qual = quality(corr, comp)
    return corr, comp, qual


def correctness_completeness_quality(pred_s, gt_s, slack):

    TP_g, TP_p, FN, FP = relaxed_confusion_matrix(pred_s, gt_s, slack)
    corr, comp, qual = compute_scores(TP_g, TP_p, FN, FP)

    return corr, comp, qual


def normalize_img(img, _min=0, _max=1):
    return cv2.normalize(img.copy(), None, alpha=_min, beta=_max, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


def read_pkl(filename):
    file = open(filename, "rb")
    output = pickle.load(file)
    file.close()
    return output


def calculate_epsilon(steps_done, egreedy_final, egreedy, egreedy_decay):
    epsilon = egreedy_final + (egreedy - egreedy_final) * \
        math.exp(-1. * steps_done / egreedy_decay)
    return epsilon


def save_model(model, file2save):
    torch.save(model.state_dict(), file2save)


def gaussian_blur(img, k=7):
    return cv2.GaussianBlur(img.astype(float), (k, k), 0)


def add_noise(img, noise_type):
    if noise_type == "gauss":
        row, col = img.shape
        mean = 0
        var = 0.005
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = img + gauss
        return noisy
    elif noise_type == "s&p":
        row, col = img.shape
        s_vs_p = 0.5
        amount = 0.03
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in img.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row, col = img.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = img + img * gauss
        return noisy


def conv_size(W, k, p, s):
    return int((W+2*p - 1*(k-1) - 1)/s+1)


def conv_output_shape(img_shape, k1, s1, k2, s2, k3, s3):
    return conv_size(conv_size(conv_size(img_shape, k1, 0, s1), k2, 0, s2), k3, 0, s3)


class NeuralNetwork(nn.Module):

    def __init__(self, img_shape, n_channels, hidden_layer, numb_outputs, normalize_image):
        super(NeuralNetwork, self).__init__()

        self.normalize_image = normalize_image
        k1, s1 = 8, 4
        self.conv1 = nn.Conv2d(in_channels=n_channels,
                               out_channels=32, kernel_size=k1, stride=s1)
        k2, s2 = 4, 2
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2)
        k3, s3 = 3, 1
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # output shape of a 50x50 image after the convolutions : 2*2*64
        conv_shape = conv_output_shape(
            img_shape=img_shape, k1=k1, s1=s1, k2=k2, s2=s2, k3=k3, s3=s3)
        if conv_shape < 1:
            raise Exception(
                'Input image shape resulted in 0-shaped convolution output (adapt the NN convolutions in consequence)')
        conv_shape = conv_shape*conv_shape*64

        self.value1 = nn.Linear(conv_shape, hidden_layer)
        self.value2 = nn.Linear(hidden_layer, 1)

        self.advantage1 = nn.Linear(conv_shape, hidden_layer)
        self.advantage2 = nn.Linear(hidden_layer, numb_outputs)

        self.activation = nn.ReLU()

    def forward(self, x):
        if self.normalize_image:
            x /= x.max()

        output_conv = self.conv1(x)
        output_conv = self.activation(output_conv)

        output_conv = self.conv2(output_conv)
        output_conv = self.activation(output_conv)

        output_conv = self.conv3(output_conv)
        output_conv = self.activation(output_conv)

        output_lin = output_conv.view(output_conv.size(0), -1)

        output_advantage = self.advantage1(output_lin)
        output_advantage = self.activation(output_advantage)
        output_advantage = self.advantage2(output_advantage)

        output_value = self.value1(output_lin)
        output_value = self.activation(output_value)
        output_value = self.value2(output_value)

        output_final = output_value + output_advantage - output_advantage.mean()
        return output_final


def preprocess_frame(frame, device):
    frame = frame.transpose((2, 0, 1))
    frame = torch.from_numpy(frame)
    frame = frame.to(device, dtype=torch.float32)
    frame = frame.unsqueeze(0)
    return frame


def plot_reward(rewards):
    plt.figure(figsize=(12, 5))
    plt.title("Rewards")
    plt.plot(rewards, alpha=0.6, color='red')
    # plt.savefig("Rewards.png")
    plt.show()
    plt.close()


def add_input_noise(imgs):

    noisy_imgs = []
    noises = ['gauss', 's&p', 'poisson']

    for img in imgs:
        noise = 'poisson'  # random.choice(noises)
        noisy_imgs.append(add_noise(img, noise))

    return noisy_imgs


def add_padding(img, pts, pad):

    return np.pad(img, pad), [np.array(pt)+np.array([pad, pad]) for pt in pts]


"""
def calculate_update_freq(steps_done):
    update_freq = update_target_freq_final - (update_target_freq_final - update_target_freq) * \
              math.exp(-1. * steps_done / update_target_freq_decay )
    return update_freq
"""
