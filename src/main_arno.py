from auto_encoder import AutoEncoder
from trainer import Trainer
from evaluator import Evaluator
# from cifar10_dataset import Cifar10Dataset
# from Dataprocessing.arno_dataset_vae import ArnoDataset
from configuration import Configuration

import torch
import torch.optim as optim
import os
import argparse
import time
import datetime
import sys
from vae_arno_dataset import ArnoDataset

# sys.path.insert(0, '/home/ubuntu/pycharm/arno-victor/Dataprocessing')
# from arno_dataset_vae import ArnoDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', nargs='?', default=Configuration.default_batch_size, type=int,
                        help='The size of the batch during training')
    parser.add_argument('--num_training_updates', nargs='?', default=Configuration.default_num_training_updates,
                        type=int, help='The number of updates during training')
    parser.add_argument('--num_hiddens', nargs='?', default=Configuration.default_num_hiddens, type=int,
                        help='The number of hidden neurons in each layer')
    parser.add_argument('--num_residual_hiddens', nargs='?', default=Configuration.default_num_residual_hiddens,
                        type=int, help='The number of hidden neurons in each layer within a residual block')
    parser.add_argument('--num_residual_layers', nargs='?', default=Configuration.default_num_residual_layers, type=int,
                        help='The number of residual layers in a residual stack')
    parser.add_argument('--embedding_dim', nargs='?', default=Configuration.default_embedding_dim, type=int,
                        help='Representing the dimensionality of the tensors in the quantized space')
    parser.add_argument('--num_embeddings', nargs='?', default=Configuration.default_num_embeddings, type=int,
                        help='The number of vectors in the quantized space')
    parser.add_argument('--commitment_cost', nargs='?', default=Configuration.default_commitment_cost, type=float,
                        help='Controls the weighting of the loss terms')
    parser.add_argument('--decay', nargs='?', default=Configuration.default_decay, type=float,
                        help='Decay for the moving averages (set to 0.0 to not use EMA)')
    parser.add_argument('--learning_rate', nargs='?', default=Configuration.default_learning_rate, type=float,
                        help='The learning rate of the optimizer during training updates')
    parser.add_argument('--use_kaiming_normal', nargs='?', default=Configuration.default_use_kaiming_normal, type=bool,
                        help='Use the weight normalization proposed in [He, K et al., 2015]')
    parser.add_argument('--unshuffle_dataset', default=not Configuration.default_shuffle_dataset, action='store_true',
                        help='Do not shuffle the dataset before training')
    parser.add_argument('--data_path', nargs='?', default='../../datasets/cifar-10-batches-py', type=str, help='The path of the data directory')
    parser.add_argument('--results_path', nargs='?', default='results', type=str,
                        help='The path of the results directory')
    parser.add_argument('--loss_plot_name', nargs='?', default='loss.png', type=str,
                        help='The file name of the training loss plot')
    parser.add_argument('--model_name', nargs='?', default='model.pth', type=str, help='The file name of trained model')
    parser.add_argument('--original_images_name', nargs='?', default='original_images.png', type=str,
                        help='The file name of the original images used in evaluation')
    parser.add_argument('--validation_images_name', nargs='?', default='validation_images.png', type=str,
                        help='The file name of the reconstructed images used in evaluation')
    args = parser.parse_args()

    # Dataset and model hyperparameters
    configuration = Configuration.build_from_args(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if cuda is available

    # Set the result path and create the directory if it doesn't exist

    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.results_path += "_" + date
    results_path = '..' + os.sep + args.results_path
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    dataset_path = '..' + os.sep + args.data_path

    # dataset = Cifar10Dataset(configuration.batch_size, dataset_path,
    #                           configuration.shuffle_dataset)  # Create an instance of CIFAR10 dataset

    print(f"batch_size: {configuration.batch_size}")

    dataset = ArnoDataset(batch_size=configuration.batch_size,
                          path="../../../datasets/arno_v1",
                          shuffle_dataset=configuration.shuffle_dataset,
                          num_workers=6,
                          im_size=128)

    auto_encoder = AutoEncoder(device, configuration).to(device)  # Create an AutoEncoder model using our GPU device

    optimizer = optim.Adam(auto_encoder.parameters(), lr=configuration.learning_rate,
                           amsgrad=True)  # Create an Adam optimizer instance
    trainer = Trainer(device, auto_encoder, optimizer, dataset)  # Create a trainer instance
    trainer.train(configuration.num_training_updates, results_path, args)  # Train our model on the CIFAR10 dataset
    auto_encoder.save(results_path + os.sep + args.model_name)  # Save our trained model
    trainer.save_loss_plot(results_path + os.sep + args.loss_plot_name)  # Save the loss plot

    evaluator = Evaluator(device, auto_encoder, dataset)  # Create en Evaluator instance to evaluate our trained model
    evaluator.reconstruct()  # Reconstruct our images from the embedded space
    evaluator.save_original_images_plot(
        results_path + os.sep + args.original_images_name)  # Save the original images for comparaison purpose
    evaluator.save_validation_reconstructions_plot(
        results_path + os.sep + args.validation_images_name)  # Reconstruct the decoded images and save them
