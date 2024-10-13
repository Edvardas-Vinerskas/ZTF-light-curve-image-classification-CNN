import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
import pandas as pd

"""import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import io
from bs4 import BeautifulSoup"""
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

import ast
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors


#main function for plotting light curve images
def lightkurve_plot_axis_on(dataframe, save_location):
  not_list_counter = 0
  for id in range(len(dataframe['objectid'])): #note that the id here is just the row number and has no meaning that is of any relevance to use
    object_data = dataframe.iloc[id, :]
    if isinstance(object_data['hmjd'], str): #the tuples from parquets were strings so we need to evaluate them
      time_data = ast.literal_eval(object_data['hmjd'])
      mag_data = ast.literal_eval(object_data['mag'])
      catflags = ast.literal_eval(object_data['catflags'])
    else:
      time_data = object_data['hmjd']
      mag_data = object_data['mag']
      catflags = object_data['catflags']


    catflag_index_list = []
    if isinstance(catflags, int):
      catflags=[catflags]
      not_list_counter += 1
      print("the catflag of this id {} is not a list lmao, counter {}".format(id, not_list_counter))
    for i in range(len(catflags)):
      if catflags[i] >= 32768:
        catflag_index_list.append(i)
    time_data = np.delete(time_data, catflag_index_list)
    mag_data = np.delete(mag_data, catflag_index_list)
    if len(mag_data) <= 10:
      continue
    title = object_data['objectid']
    fig = plt.figure(figsize=(4, 4)) #changed to 4x4 so that it would be symmetric
    plt.scatter(time_data, mag_data, s=(72. / fig.dpi) ** 2, marker=',')
    plt.xlabel('hmjd')
    plt.ylabel('magnitude')
    plt.title('{}'.format(title))
    plt.savefig('{}/{}.jpg'.format(save_location, title), dpi=100) #changed to jpg
    plt.close()


#function for splitting arrays (i.e. images) into 5 equal parts
def split_array(arr):

  # Calculate the length of each part
  part_length = len(arr) // 5

  # Use array_split to create 5 equal parts
  parts = np.array_split(arr, 5)

  return parts


#function for plotting images sliced into 5 parts
def lightkurve_plot_axis_on_cutup(dataframe, save_location):
  not_list_counter = 0
  for id in range(len(dataframe['objectid'])): #note that the id here is just the row number and has no meaning that is of any relevance
    object_data = dataframe.iloc[id, :]
    if isinstance(object_data['hmjd'], str): #the tuples from parquets were strings so we need to evaluate them
      time_data = ast.literal_eval(object_data['hmjd'])
      mag_data = ast.literal_eval(object_data['mag'])
      catflags = ast.literal_eval(object_data['catflags'])
    else:
      time_data = object_data['hmjd']
      mag_data = object_data['mag']
      catflags = object_data['catflags']


    catflag_index_list = []
    if isinstance(catflags, int):
      catflags=[catflags]
      not_list_counter += 1
      print("the catflag of this id {} is not a list lmao, counter {}".format(id, not_list_counter))
    for i in range(len(catflags)):
      if catflags[i] >= 32768:
        catflag_index_list.append(i)
    time_data = np.delete(time_data, catflag_index_list)
    mag_data = np.delete(mag_data, catflag_index_list)
    if len(mag_data) <= 10:
      continue

    #we cut up the time and magnitude data

    if len(mag_data) >= 100:
      time_data_result = split_array(time_data)
      mag_data_result = split_array(mag_data)
      for i in range(len(mag_data_result)):
        title = object_data['objectid']
        fig = plt.figure(figsize=(4, 4)) #changed to 4x4 so that it would be symmetric
        plt.scatter(time_data_result[i], mag_data_result[i], s=(72. / fig.dpi) ** 2, marker=',')
        plt.xlabel('hmjd')
        plt.ylabel('magnitude')
        plt.title('{}'.format(title))
        plt.savefig('{}/{}_{}.jpg'.format(save_location, title, i+1), dpi=100) #changed to jpg
        plt.close()
    else:
      title = object_data['objectid']
      fig = plt.figure(figsize=(4, 4))  # changed to 4x4 so that it would be symmetric
      plt.scatter(time_data, mag_data, s=(72. / fig.dpi) ** 2, marker=',')
      plt.xlabel('hmjd')
      plt.ylabel('magnitude')
      plt.title('{}'.format(title))
      plt.savefig('{}/{}.jpg'.format(save_location, title), dpi=100)  # changed to jpg
      plt.close()


#function to try to estimate local density
def local_density(points, k=50):
  nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(points)
  distances, _ = nbrs.kneighbors(points)
  density = 1 / np.mean(distances, axis=1)
  return density



def lightkurve_plot_axis_on_endtofront(dataframe, save_location):
  not_list_counter = 0
  for id in range(len(dataframe['objectid'])): #note that the id here is just the row number and has no meaning that is of any relevance to use
    object_data = dataframe.iloc[id, :]
    if isinstance(object_data['hmjd'], str): #the tuples from parquets were strings so we need to evaluate them
      time_data = ast.literal_eval(object_data['hmjd'])
      mag_data = ast.literal_eval(object_data['mag'])
      catflags = ast.literal_eval(object_data['catflags'])
    else:
      time_data = object_data['hmjd']
      mag_data = object_data['mag']
      catflags = object_data['catflags']

    #turn everything into numpy arrays so that it works (if it is already an array then nothing should happen)
    time_data = np.array(time_data)
    mag_data = np.array(mag_data)
    catflags = np.array(catflags)

    # we splice here (attaching the back to the front)

    #check if we have an int on our hands (and if we do escape the loop)

    #appending the end to the front part
    if np.size(mag_data) >= 4:
      difference_end = time_data[len(time_data) - int(0.3 * len(time_data))] - time_data[0]
      difference_front = time_data[-1] - time_data[len(time_data) - int(0.3 * len(time_data))]
      time_data_front = time_data[:len(time_data) - int(0.3 * len(time_data))] + difference_front
      time_data_end = time_data[-int(0.3 * len(time_data)):] - difference_end
      time_data = np.concatenate((time_data_end, time_data_front))

      mag_data_front = mag_data[:len(mag_data) - int(0.3 * len(mag_data))]
      mag_data_end = mag_data[-int(0.3 * len(mag_data)):]
      mag_data = np.concatenate((mag_data_end, mag_data_front))

      catflags_front = catflags[:len(catflags) - int(0.3 * len(catflags))]
      catflags_end = catflags[-int(0.3 * len(catflags)):]
      catflags = np.concatenate((catflags_end, catflags_front))
    else:
      continue

    catflag_index_list = []
    if isinstance(catflags, int):
      catflags=[catflags]
      not_list_counter += 1
      print("the catflag of this id {} is not a list lmao, counter {}".format(id, not_list_counter))
    for i in range(len(catflags)):
      if catflags[i] >= 32768:
        catflag_index_list.append(i)
    time_data = np.delete(time_data, catflag_index_list)
    mag_data = np.delete(mag_data, catflag_index_list)
    if len(mag_data) <= 10:
      continue


    title = object_data['objectid']
    fig = plt.figure(figsize=(4, 4)) #changed to 4x4 so that it would be symmetric
    plt.scatter(time_data, mag_data, s=(72. / fig.dpi) ** 2, marker=',')
    plt.xlabel('hmjd')
    plt.ylabel('magnitude')
    plt.title('{}'.format(title))
    # plt.axis("off")
    plt.savefig('{}/{}_endfront.jpg'.format(save_location, title), dpi=100) #changed to jpg
    print("saved image {}".format(id))
    plt.close()


#function for training the CNN
def train_loop(our_data, model, loss_fn, optimizer, threshold, device):
  size = len(our_data.dataset) #gets the number of samples in the dataset
  scaler = GradScaler() #recommended by claude to increase gpu utilisation
  num_batches = len(our_data)
  test_loss, total_correct, total_labels = 0, 0, 0
  #set the model to training mode - important for batch normalization and dropout layers
  model.train()
  for batch, (X, y) in enumerate(our_data): #X is data, y the correct labels
    X, y = X.to(device), y.to(device) #move input and labels to device

    #autocast should help with effective gpu utilisation
    with autocast():
      pred = model(X)
      loss = loss_fn(pred, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()  # resets the gradients to 0 to prevent accumulation (zero clue)
    test_loss += loss.item()
    predictions = (torch.sigmoid(pred) > threshold).float()
    total_correct += (predictions == y).float().sum().item()  # sums up the total correct predictions
    total_labels += y.numel()  # sums up the total number of labels

  test_loss /= num_batches
  accuracy_avg = total_correct / total_labels

  model_evaluation_list_tags = ["total_accuracy_train", "Avg_loss_train"]
  model_evaluation_list = [100*accuracy_avg, test_loss]
  model_evaluation_dict = {}
  for i in range(len(model_evaluation_list)):
      model_evaluation_dict[model_evaluation_list_tags[i]] = model_evaluation_list[i]
  return model_evaluation_dict


def train_loop_optuna(our_data, model, loss_fn, optimizer, device_id):
  size = len(our_data.dataset) #gets the number of samples in the dataset
  scaler = GradScaler()
  num_batches = len(our_data)
  #set the model to training mode - important for batch normalization and dropout layers
  model.train()
  for batch, (X, y) in enumerate(our_data): #X is data, y the correct labels
    X, y = X.to(device_id), y.to(device_id) #move input and labels to device

    with autocast():
        pred = model(X)
        # pred = pred.squeeze() #this line is forthe binary classificator to reduce the dimensionality of the output
        loss = loss_fn(pred, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()

#function for testing the CNN
def test_loop(our_data, model, batch_size, loss_fn, trainingcsvpath, save_file, threshold, device): #test and validation loop are the same innit
  #set the model to evaluation mode - important for batch normalization and dropout layers
  model.eval()
  size = len(our_data.dataset) #the number of samples in the dataset
  num_batches = len(our_data) #number of batches in the our_data
  test_loss, accuracy_total, label_counter, total_precision, total_recall = 0, 0, 0, 0, 0
  total_correct, total_labels, true_positive, false_positive, false_negative, true_negative = 0, 0, 0, 0, 0, 0
  label_list_acc = []
  training_df = pd.read_csv(trainingcsvpath)
  training_df_cols = training_df.columns.values.tolist()
  training_df_cols = training_df_cols[2:]
  correct_per_label = torch.zeros(len(training_df_cols), device = device)
  full_confusion_array = np.zeros((len(training_df_cols), len(training_df_cols)))
  #creates a place holder the length of the label number for each class accuracy keeping
  for i in range(len(training_df_cols)):
    label_list_acc.append(0)

  # since we are evaluating the model we do not need to compute the gradient anymore:
  with torch.no_grad(): #disables gradient computation
    for X, y in our_data:
      X, y = X.to(device), y.to(device)
      pred = model(X) #forward pass = calculates the predicted labels
      test_loss += loss_fn(pred, y).item()
      predictions = (torch.sigmoid(pred) > threshold).float() #we need to normalise the logits using softmax/sigmoid since now we are operating outside the cross entropy function

      correct_per_label += (predictions == y).float().sum(dim=0)
      label_counter += y.numel()/len(training_df_cols)

      total_correct += (predictions == y).float().sum().item() #sums up the total correct predictions
      total_labels += y.numel() #sums up the total number of labels

      #TP, FN, FP, TN
      true_positive += ((predictions == y) & (y == 1)).sum(dim=0)
      false_negative += ((predictions != y) & (y == 1)).sum(dim=0)
      false_positive += ((predictions != y) & (y == 0)).sum(dim=0)
      true_negative += ((predictions == y) & (y == 0)).sum(dim=0)

      """confusion_array = []
      for jj in range(len(training_df_cols)):
        label_confusion_jj = []
        for j in range(len(training_df_cols)):
          label_matrix_jj_j = []
          for i in range(pred.size()[0]):
            jj_when_j = ((predictions[i, jj] == 1) & (y[i, j] == 1)).float().item()
            label_matrix_jj_j.append(jj_when_j)  # appends individual results from each tensor
          label_confusion_jj.append(sum(label_matrix_jj_j))  # the value when jj is predicted and j is present summed up over the batch (what we want)
        confusion_array.append(label_confusion_jj)  # here we get a list of lists with each list being a single label being predicted when some other label was present
      confusion_array = np.array(confusion_array)
      full_confusion_array += confusion_array"""

  #total precision (per class)
  total_precision = torch.div(true_positive, torch.add(true_positive, false_positive))
  #total recall
  total_recall = torch.div(true_positive, torch.add(true_positive, false_negative))

  test_loss /= num_batches
  accuracy_avg = total_correct / total_labels
  #calculates the accuracy for each individual label
  label_list_acc = torch.div(correct_per_label, label_counter)

  with open('{}.txt'.format(save_file), mode='a') as file:
    file.write(f"Test Error: \n Accuracy: {(100*accuracy_avg):>0.1f}%, Avg loss: {test_loss:>8f}\n Precision: {total_precision}\n Recall: {total_recall} \n")
    for i in range(len(training_df_cols)):
      file.write("For label \n {}: precision {}, recall {} \n".format(training_df_cols[i], total_precision[i], total_recall[i]))



  #so now we just want to pull out the TP, TN, FP, FN above and plot them for each class
  model_evaluation_list_tags = ["total_accuracy", "Avg_loss", "precision", "recall", "labels", "TP", "FP", "FN", "TN"]#, "FULL_CONFUSION"]
  model_evaluation_list = [100*accuracy_avg, test_loss, total_precision, total_recall, label_list_acc, true_positive, false_positive, false_negative, true_negative]#, full_confusion_array]
  model_evaluation_dict = {}
  for i in range(len(model_evaluation_list)):
      model_evaluation_dict[model_evaluation_list_tags[i]] = model_evaluation_list[i]
  return model_evaluation_dict, training_df_cols


def test_loop_optuna(our_data, model, batch_size, loss_fn, trainingcsvpath, threshold, device_id):
  #set the model to evaluation mode - important for batch normalization and dropout layers
  model.eval()
  size = len(our_data.dataset) #the number of samples in the dataset
  num_batches = len(our_data) #number of batches in the our_data
  test_loss, accuracy_total, label_counter, total_precision, total_recall = 0, 0, 0, 0, 0
  total_correct, total_labels, true_positive, false_positive, false_negative, true_negative = 0, 0, 0, 0, 0, 0
  label_list_acc = []  #possibly not useful anymore
  training_df = pd.read_csv(trainingcsvpath)
  training_df_cols = training_df.columns.values.tolist()
  training_df_cols = training_df_cols[2:]
  correct_per_label = torch.zeros(len(training_df_cols), device = device_id)
  full_confusion_array = np.zeros((len(training_df_cols), len(training_df_cols)))
  #creates a place holder the length of the label number for each class accuracy keeping
  for i in range(len(training_df_cols)):
    label_list_acc.append(0)

  # since we are evaluating the model we do not need to compute the gradient anymore:
  with torch.no_grad(): #disables gradient computation
    for X, y in our_data:
      X, y = X.to(device_id), y.to(device_id)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      #Convert probabilities to binary predictions using a threshold of 0.7
      predictions = (torch.sigmoid(pred) > threshold).float() #we need to normalise the logits using softmax/sigmoid since now we are operating outside the cross entropy function

      correct_per_label += (predictions == y).float().sum(dim=0)
      label_counter += y.numel()/len(training_df_cols)

      total_correct += (predictions == y).float().sum().item() #sums up the total correct predictions
      total_labels += y.numel() #sums up the total number of labels

      #TP, FN, FP, TN
      true_positive += ((predictions == y) & (y == 1)).sum(dim=0)
      false_negative += ((predictions != y) & (y == 1)).sum(dim=0)
      false_positive += ((predictions != y) & (y == 0)).sum(dim=0)
      true_negative += ((predictions == y) & (y == 0)).sum(dim=0)

  # Gather metrics from all processes
  metrics = torch.tensor([test_loss, total_correct, total_labels]).to(device_id)
  dist.all_reduce(metrics)
  test_loss, total_correct, total_labels = metrics.tolist()

  dist.all_reduce(correct_per_label)
  dist.all_reduce(true_positive)
  dist.all_reduce(false_positive)
  dist.all_reduce(false_negative)
  dist.all_reduce(true_negative)

  #total precision (per class as one would expect)
  denominator = torch.add(true_positive, false_positive)
  total_precision = torch.where(
      denominator > 0,
      torch.div(true_positive, denominator),
      torch.zeros_like(denominator)
  )
  #total recall
  total_recall = torch.div(true_positive, torch.add(true_positive, false_negative))

  test_loss = test_loss / num_batches
  accuracy_avg = total_correct / total_labels
  #calculates the accuracy for each individual label
  label_list_acc = torch.div(correct_per_label, label_counter)


  #so now we just want to pull out the TP, TN, FP, FN above and plot them for each class
  model_evaluation_dict = {
    "total_accuracy": 100 * accuracy_avg,
    "Avg_loss": test_loss,
    "precision": total_precision,
    "recall": total_recall,
    "labels": label_list_acc,
    "TP": true_positive,
    "FP": false_positive,
    "FN": false_negative,
    "TN": true_negative
  }
  return model_evaluation_dict, training_df_cols



#this is from a matplotlib library talking about heatmaps
def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                       textcolors=("black", "white"),
                       threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
      data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
      threshold = im.norm(threshold)
    else:
      threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
      valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    data_FP = np.array([["TP", 'FP'], ['FN', 'TN']])
    texts = []
    for i in range(data.shape[0]):
      for j in range(data.shape[1]):
        kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
        text = im.axes.text(j, i, "{}: {:.4g}".format(data_FP[i, j], data[i, j]), **kw)
        texts.append(text)

    return texts


def annotate_heatmap_without_TP(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
  """
  A function to annotate a heatmap.

  Parameters
  ----------
  im
      The AxesImage to be labeled.
  data
      Data used to annotate.  If None, the image's data is used.  Optional.
  valfmt
      The format of the annotations inside the heatmap.  This should either
      use the string format method, e.g. "$ {x:.2f}", or be a
      `matplotlib.ticker.Formatter`.  Optional.
  textcolors
      A pair of colors.  The first is used for values below a threshold,
      the second for those above.  Optional.
  threshold
      Value in data units according to which the colors from textcolors are
      applied.  If None (the default) uses the middle of the colormap as
      separation.  Optional.
  **kwargs
      All other arguments are forwarded to each call to `text` used to create
      the text labels.
  """

  if not isinstance(data, (list, np.ndarray)):
    data = im.get_array()

  # Normalize the threshold to the images color range.
  if threshold is not None:
    threshold = im.norm(threshold)
  else:
    threshold = im.norm(data.max()) / 2.

  # Set default alignment to center, but allow it to be
  # overwritten by textkw.
  kw = dict(horizontalalignment="center",
            verticalalignment="center")
  kw.update(textkw)

  # Get the formatter in case a string is supplied
  if isinstance(valfmt, str):
    valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

  # Loop over the data and create a `Text` for each "pixel".
  # Change the text's color depending on the data.
  texts = []
  for i in range(data.shape[0]):
    for j in range(data.shape[1]):
      kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
      text = im.axes.text(j, i, "{:.4g}".format(data[i, j]), **kw)
      texts.append(text)

  return texts