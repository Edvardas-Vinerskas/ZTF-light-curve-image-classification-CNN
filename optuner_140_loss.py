

import argparse
from functools import partial
import os
import scipy
import sklearn

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler
#from torchvision.io import read_image--
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
#from torchvision import datasets, transforms
import torchvision.models as models
#from torchvision.transforms import ToTensor
#from torchvision.transforms import Lambda
from torch import nn

from data_class import ZTF_lightkurve_img
from functions import train_loop_optuna, test_loop_optuna, test_loop

import optuna
from optuna.integration import TorchDistributedTrial

from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline


def setup(backend, rank, world_size, master_port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port #12355
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"Using {backend} backend.")


def cleanup():
    dist.destroy_process_group()


# split the dataset into training and validation (test comes last and outside the training loop)
training_csv_path = "annotations/training_set_5_other_multiplied.csv"
annotations_dir = "annotations/training_set_5_other_multiplied.csv"
img_dir = 'lightkurve_plots/lightkurve_plots_match_multiplied'

# initialise our dataset
dataset = ZTF_lightkurve_img(annotations_file=annotations_dir,
                             img_dir=img_dir)

trainval_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.8),
                                                               len(dataset) - int(len(dataset) * 0.8)])

train_set, valid_set = torch.utils.data.random_split(trainval_set, [int(len(trainval_set) * 0.8),
                                                                    len(trainval_set) - int(
                                                                        len(trainval_set) * 0.8)])

epochs=15 #changed to 15 to reduce compute time
number_of_trials=150
num_workers = 8
loss_fn = nn.BCEWithLogitsLoss()


#device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


# Function to load the best metric from a file
def load_best_metric(filepath='optuna_models/best_model_200_loss.txt'):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            content = file.readlines()
            if content:
                return float(content[1].strip().split(': ')[1])
    return float('inf')

# Function to save the best metric to a file
def save_best_metric(filepath='optuna_models/best_model_200_loss.txt', trial_number=None, metric=None,
                     loss=None):
    with open(filepath, 'w') as file:
        file.write(f"Trial number of best model: {trial_number}\n")
        file.write(f"Best combined metric: {metric}\n")
        file.write(f"Best loss: {loss}\n")


#define a function to be minimised
def objective(single_trial, device_id):
    trial = TorchDistributedTrial(single_trial)

    ZTF_model = models.resnet18(pretrained=False)
    ZTF_model.fc = torch.nn.Linear(ZTF_model.fc.in_features, 31)
    ZTF_model = ZTF_model.to(device_id) #so do I but the model onto the device or rank?
    ZTF_model = DDP(ZTF_model, device_ids=None if device_id == 'cpu' else [device_id])

    #generate hyperparameters for objective to try
    #the first string should just act as a label
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "Adagrad"])
    lr = trial.suggest_float("lr", 0.00001, 0.1, log=True)
    threshold = trial.suggest_float("threshold", 0.5, 0.9)
    batch_size = trial.suggest_int("batch_size", 10, 200, log=True)

    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.9)  # Suggest a momentum value between 0 and 0.9
        nesterov = trial.suggest_categorical("nesterov", [False, True])  # Suggest whether to use Nesterov momentum

        # Initialize SGD with suggested momentum and Nesterov options
        optimizer = torch.optim.SGD(ZTF_model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)
    else:
        # Initialize the selected optimizer without momentum settings
        optimizer = getattr(torch.optim, optimizer_name)(ZTF_model.parameters(), lr=lr)

    #parallelisation code from claude.ai
    train_sampler = DistributedSampler(train_set)
    valid_sampler = DistributedSampler(valid_set)

    # load the data
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)

    print("starting trial {}".format(trial.number))
    #train the model here
    for t in range(epochs):
        #shuffles the data (we cannot use shuffle in DataLoader)
        train_loader.sampler.set_epoch(t)
        train_loop_optuna(train_loader, ZTF_model, loss_fn, optimizer,
                                                 device_id)
        model_evaluation_dict, training_df_cols = test_loop_optuna(valid_loader, ZTF_model, batch_size, loss_fn,
                                                            training_csv_path, threshold, device_id)
        loss_score = model_evaluation_dict["Avg_loss"]

        combined_metric = loss_score


        if dist.get_rank() == 0:
            best_val_loss = load_best_metric()
            if combined_metric < best_val_loss:
                torch.save(ZTF_model.state_dict(), 'optuna_models/ZTF_model_optuna_200_loss.pth')
                save_best_metric('optuna_models/best_model_200_loss.txt', trial_number=trial.number,
                                 metric=combined_metric, loss = loss_score)


        trial.report(combined_metric, t)
        if trial.should_prune(): #default pruning is MedianPruner but SuccessiveHalvingPruner (what I use) is recommended by their 003 tutorial
            raise optuna.exceptions.TrialPruned()

    print("returning scores of trial {}".format(trial.number))
    #loss function here only
    return combined_metric


#claude modified for parallelisation
#https://github.com/optuna/optuna-examples/blob/main/pytorch/pytorch_distributed_spawn.py


def run_optimize(rank, world_size, device_ids, return_dict, master_port):
    device = "cpu" if len(device_ids) == 0 else device_ids[rank]
    print(f"Running basic DDP example on rank {rank} device {device}.")

    # Set environmental variables required by torch.distributed.
    backend = "gloo"
    if torch.distributed.is_nccl_available():
        if device != "cpu":
            backend = "nccl"
    setup(backend, rank, world_size, master_port)

    if rank == 0:
        study = optuna.create_study(direction="minimize",
                                pruner=optuna.pruners.SuccessiveHalvingPruner(),
                                sampler=optuna.samplers.TPESampler(n_startup_trials=10, n_ei_candidates=24))
        study.optimize(
            partial(objective, device_id=device),
            n_trials=number_of_trials
        )
        return_dict["study"] = study
    else:
        for _ in range(number_of_trials):
            try:
                objective(None, device)
            except optuna.TrialPruned:
                pass

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PyTorch distributed data-parallel training with spawn example."
    )
    parser.add_argument(
        "--device-ids",
        "-d",
        nargs="+",
        type=int,
        default=[0],
        help="Specify device_ids if using GPUs.",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disable CUDA training."
    )
    parser.add_argument("--master-port", type=str, default="12355", help="Specify port number.")
    args = parser.parse_args()
    if args.no_cuda:
        device_ids = []
    else:
        device_ids = args.device_ids

    world_size = max(len(device_ids), 1)
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(
        run_optimize,
        args=(world_size, device_ids, return_dict, args.master_port),
        nprocs=world_size,
        join=True,
    )
    study = return_dict["study"]

    pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

    print("Study statistics: ")
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Number of pruned trials: {}".format(len(pruned_trials)))
    print("Number of complete trials: {}".format(len(complete_trials)))

    print("Best trial:")
    trial = study.best_trial

    print("Value: ", trial.value)

    print("Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

    with open("optuna_models/study_stats_200_loss.txt", 'w') as file:
        file.write("Study statistics: \n")
        file.write("Number of finished trials: {}\n".format(len(study.trials)))
        file.write("Number of pruned trials: {}\n".format(len(pruned_trials)))
        file.write("Number of complete trials: {}\n".format(len(complete_trials)))
        file.write("Best trial:\n")
        file.write("Value: {}\n".format(trial.value))
        file.write("Params: \n")
        for key, value in trial.params.items():
            file.write(" \n{}: {}".format(key, value))

    # Save results to csv file  (https://github.com/elena-ecn/optuna-optimization-for-PyTorch-CNN/blob/main/optuna_optimization.py)
    df = study.trials_dataframe()
    df = df.loc[df['state'] == 'COMPLETE']  # Keep only results that did not prune
    # df = df.drop('state', axis=1)                 # Exclude state column
    df = df.sort_values('value')  # Sort based on accuracy
    df.to_csv('optuna_models/optuna_results_200_loss.csv', index=False)  # Save to csv file

    sizee_1 = (10, 6)
    sizee_2 = (20, 6)
    sizee_3 = (20, 12)
    try:
        plt.rcdefaults()
        fig_optimization_history = plot_optimization_history(study)
        plt.tight_layout()
        fig_optimization_history = fig_optimization_history.figure
        fig_optimization_history.set_size_inches(sizee_1)
        fig_optimization_history.savefig("optuna_plots/optimization_history_200_loss.png", dpi=300)
        plt.close()
    except Exception as e:
        print("Error saving optimization history plot: {e}")

    try:
        # Plots intermediate values vs epochs
        plt.rcdefaults()
        fig_intermediate_values = plot_intermediate_values(study)
        plt.tight_layout()
        fig_intermediate_values = fig_intermediate_values.figure
        fig_intermediate_values.set_size_inches(sizee_1)
        fig_intermediate_values.savefig("optuna_plots/intermediate_values_200_loss.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error saving intermediate values plot: {e}")

    try:
        # Plot the high-dimensional parameter relationships in a study with Matplotlib
        plt.rcdefaults()
        fig_parallel_coordinate = plot_parallel_coordinate(study)
        plt.tight_layout()
        fig_parallel_coordinate = fig_parallel_coordinate.figure
        fig_parallel_coordinate.set_size_inches(sizee_2)
        fig_parallel_coordinate.savefig("optuna_plots/parallel_coordinate_200_loss.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error saving parallel coordinate plot: {e}")

    try:
        # Plot the parameter relationship as contour plot in a study (Fedor plot?)
        plt.rcdefaults()
        fig_contour = optuna.visualization.matplotlib.plot_contour(study)
        fig_contour = fig_contour[0, 0].figure
        fig_contour.set_size_inches(sizee_3)
        fig_contour.savefig("optuna_plots/contour_200_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error saving contour plot: {e}")

    try:
        # Plot the parameter relationship as slice plot in a study
        plt.rcdefaults()
        fig_slice = optuna.visualization.matplotlib.plot_slice(study)
        fig_slice = fig_slice[0].figure
        fig_slice.set_size_inches(sizee_3)
        fig_slice.savefig("optuna_plots/slice_200_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error saving slice plot: {e}")

    try:
        # plots how important are our parameters
        plt.rcdefaults()
        fig_param_importances = plot_param_importances(study)
        plt.tight_layout()
        fig_param_importances = fig_param_importances.figure
        fig_param_importances.set_size_inches(sizee_1)
        fig_param_importances.savefig("optuna_plots/param_importances_200_loss.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error saving parameter importances plot: {e}")

    try:
        # Plot the timeline of a study (not very useful but why not)
        plt.rcdefaults()
        fig_timeline = plot_timeline(study)
        plt.tight_layout()
        fig_timeline = fig_timeline.figure
        fig_timeline.set_size_inches(sizee_1)
        fig_timeline.savefig("optuna_plots/timeline_200_loss.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error saving timeline plot: {e}")

    # test out the best hyperparameters on the test set
    test_loader = DataLoader(test_set, batch_size=trial.params["batch_size"], num_workers=num_workers)

    # need to somehow save the weights of the best model
    gpu_id = 0
    devicers = torch.device(f'cuda:{gpu_id}')

    ZTF_model_best = models.resnet18()
    ZTF_model_best.fc = torch.nn.Linear(ZTF_model_best.fc.in_features, 31)
    state_dict = torch.load("optuna_models/ZTF_model_optuna_200_loss.pth")
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    ZTF_model_best.load_state_dict(new_state_dict)
    ZTF_model_best = ZTF_model_best.to(devicers)


    xx, yyy = test_loop(test_loader, ZTF_model_best, trial.params["batch_size"], loss_fn,
                        training_csv_path, 'optuna_models/best_model_score_200_loss', devicers)

    with open('optuna_models/ZTF_model_test_200_loss.txt', 'w') as file:
        file.write("Accuracy of best model: {}\n".format(xx["total_accuracy"]))
        file.write("Loss of best model: {}\n".format(xx["Avg_loss"]))
        file.write("Precision of best model: \n {}\n".format(xx["precision"]))
        file.write("Recall of best model: \n {}\n".format(xx["recall"]))