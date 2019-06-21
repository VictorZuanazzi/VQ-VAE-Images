#####################################################################################
# MIT License                                                                       #
#                                                                                   #
# Copyright (C) 2019 Charly Lamothe                                                 #
# Copyright (C) 2018 Zalando Research                                               #
#                                                                                   #
# This file is part of VQ-VAE-images.                                               #
#                                                                                   #
#   Permission is hereby granted, free of charge, to any person obtaining a copy    #
#   of this software and associated documentation files (the "Software"), to deal   #
#   in the Software without restriction, including without limitation the rights    #
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
#   copies of the Software, and to permit persons to whom the Software is           #
#   furnished to do so, subject to the following conditions:                        #
#                                                                                   #
#   The above copyright notice and this permission notice shall be included in all  #
#   copies or substantial portions of the Software.                                 #
#                                                                                   #
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
#   SOFTWARE.                                                                       #
#####################################################################################

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import time
import datetime

from evaluator import Evaluator


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


class Trainer(object):

    def __init__(self, device, model, optimizer, dataset, verbose=True):
        self._device = device
        self._model = model
        self._optimizer = optimizer
        self._dataset = dataset
        self._verbose = verbose
        self._train_res_recon_error = []
        self._train_res_perplexity = []

    def train(self, num_training_updates, results_path, args):
        self._model.train()

        global_time = time.time()

        iterator = iter(cycle(self._dataset.training_loader))
        for i in range(num_training_updates):
            try:
                (data, _) = next(iterator)
            except:
                data = next(iterator)
            data = data.to(self._device)
            self._optimizer.zero_grad()

            """
            The perplexity a useful value to track during training.
            It indicates how many codes are 'active' on average.
            """
            vq_loss, data_recon, perplexity = self._model(data.view(-1,3, 128, 128))
            recon_error = torch.mean((data_recon - data) ** 2) / self._dataset.train_data_variance
            loss = recon_error + vq_loss
            loss.backward()

            self._optimizer.step()

            self._train_res_recon_error.append(recon_error.item())
            self._train_res_perplexity.append(perplexity.item())

            if self._verbose and (i % int(num_training_updates / 1000) == 0):
                print('Iteration #{}'.format(i + 1))

                # calculate the elapsed time and estimation to completion
                elapsed = time.time() - global_time
                etc = elapsed*(num_training_updates - i + 1)/(i + 1)
                elapsed = str(datetime.timedelta(seconds=elapsed))
                etc = str(datetime.timedelta(seconds=etc))

                print(f"[{elapsed} / {etc}] Reconstruction error: {np.mean(self._train_res_recon_error[-100:]):.3f} "
                      f"Perplexity: {np.mean(self._train_res_perplexity[-100:]):.3f}\n")

                # I want to generate images to see the progress
                evaluator = Evaluator(self._device, self._model,
                                      self._dataset)  # Create en Evaluator instance to evaluate our trained model
                evaluator.reconstruct()  # Reconstruct our images from the embedded space
                evaluator.save_original_images_plot(
                    results_path + os.sep + str(i) + args.original_images_name)  # Save the original images for comparaison purpose
                evaluator.save_validation_reconstructions_plot(
                    results_path + os.sep + str(i) + args.validation_images_name)  # Reconstruct the decoded images and save them


    def save_loss_plot(self, path):
        maximum_window_length = 201
        train_res_recon_error_len = len(self._train_res_recon_error)
        train_res_recon_error_len = train_res_recon_error_len if train_res_recon_error_len % 2 == 1 else train_res_recon_error_len - 1
        train_res_perplexity_len = len(self._train_res_perplexity)
        train_res_perplexity_len = train_res_perplexity_len if train_res_perplexity_len % 2 == 1 else train_res_perplexity_len - 1
        polyorder = 7

        train_res_recon_error_smooth = savgol_filter(
            self._train_res_recon_error,
            maximum_window_length if train_res_recon_error_len > maximum_window_length else train_res_recon_error_len,
            polyorder
        )
        train_res_perplexity_smooth = savgol_filter(
            self._train_res_perplexity,
            maximum_window_length if train_res_perplexity_len > maximum_window_length else train_res_perplexity_len,
            polyorder
        )

        fig = plt.figure(figsize=(16, 8))

        ax = fig.add_subplot(1, 2, 1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('Iterations')

        ax = fig.add_subplot(1, 2, 2)
        ax.plot(train_res_perplexity_smooth)
        ax.set_title('Smoothed Average codebook usage (perplexity).')
        ax.set_xlabel('Iterations')

        fig.savefig(path)
        plt.close(fig)
