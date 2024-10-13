
#CODE FOR TRAINING THE CUSTOM CNN

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from data_class import ZTF_lightkurve_img
from model_skeleton import cNeuralNetwork
from functions import train_loop, test_loop, annotate_heatmap, annotate_heatmap_without_TP


learning_rate = 0.0001
batch_size = 100
epochs = 50

training_csv_path = "annotations/training_set_5_other_multiplied.csv"
annotations_dir = "annotations/training_set_5_other_multiplied.csv"
img_dir = 'lightkurve_plots/lightkurve_plots_match_multiplied'

#initialise our dataset
dataset = ZTF_lightkurve_img(annotations_file = annotations_dir, img_dir = img_dir)


print('dataset initialised')


train_set, test_set = torch.utils.data.random_split(dataset, [int(len(dataset.img_labels)*0.8), len(dataset.img_labels)-int(len(dataset.img_labels)*0.8)]) #object_id number is 170632

#load the data
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers=4, pin_memory=True)

print('data loaded')

#check if cuda is available
device = (
    'cuda'
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print('using {}'.format(device))

#initialise our model
ZTF_model = cNeuralNetwork()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(ZTF_model.parameters(), lr = learning_rate)


#data parallelisation implementation (this is supposedly for a single machine)
ZTF_model = nn.DataParallel(ZTF_model)
ZTF_model = ZTF_model.to(device)


print('starting model training')

# train the model here
epoch_list = []
model_evaluation_dict_list = []
model_evaluation_dict_train_list = []
for t in range(epochs):
    print("Epoch {}\n----------------------------------------------".format(t + 1))
    epoch_list.append(t)
    model_evaluation_dict_train = train_loop(train_loader, ZTF_model, loss_fn, optimizer, device)
    model_evaluation_dict, training_df_cols = test_loop(test_loader, ZTF_model, batch_size, loss_fn, training_csv_path, 'accuracy', device)
    model_evaluation_dict_list.append(model_evaluation_dict)
    model_evaluation_dict_train_list.append(model_evaluation_dict_train)

#save our model parameters to a file
torch.save(ZTF_model, 'models/ZTF_model.pth')

#the rest is just evaluation plots

total_accuracy_train = []
avg_loss_train = []
total_accuracy = []
avg_loss = []
precision = []
recall = []
labels = []
TP = []
FP = []
FN = []
TN = []
#FULL_CONFUSION = []
for i in range(len(model_evaluation_dict_list)):
    total_accuracy_train.append(model_evaluation_dict_train_list[i]["total_accuracy_train"])
    avg_loss_train.append(model_evaluation_dict_train_list[i]["Avg_loss_train"])
    total_accuracy.append(model_evaluation_dict_list[i]["total_accuracy"]) #should just be a nice list of 50 values
    avg_loss.append(model_evaluation_dict_list[i]["Avg_loss"])
    precision.append(model_evaluation_dict_list[i]["precision"]) #for every epoch we should get a tensor with values for each label
    recall.append(model_evaluation_dict_list[i]["recall"]) #also a list of tensors
    labels.append(model_evaluation_dict_list[i]["labels"]) #also a list of tensors
    TP.append(model_evaluation_dict_list[i]["TP"]) #a list of tensors (one tensor per epoch)
    FP.append(model_evaluation_dict_list[i]["FP"])
    FN.append(model_evaluation_dict_list[i]["FN"])
    TN.append(model_evaluation_dict_list[i]["TN"])
    #FULL_CONFUSION.append(model_evaluation_dict_list[i]["FULL_CONFUSION"]) #should be a list of arrays 28x28 with 1 array per epoch
metric_list = [total_accuracy, avg_loss, precision, recall, labels]
metric_list_train = [total_accuracy_train, avg_loss_train]

precision_dict = {}
recall_dict = {}
labels_dict ={}
for i in range(len(training_df_cols)):
    precision_dict[training_df_cols[i]] = []
    recall_dict[training_df_cols[i]] = []
    labels_dict[training_df_cols[i]] = []
    for j in range(len(precision)):
        precision_dict[training_df_cols[i]].append(precision[j][i].item())
        recall_dict[training_df_cols[i]].append(recall[j][i].item())
        labels_dict[training_df_cols[i]].append(labels[j][i].item())

#precision, recall and labels are lists of tensors
#plotting/saving some graphs for model evaluation

#title = "f score vs epoch"
model_evaluation_keys = ["total_accuracy", "Avg_loss", "precision", "recall", "labels"]
fig, axs = plt.subplots(10, 10, figsize=(70, 50), layout='constrained')
row, col = 0, 0 #change row and col inside the loops based on nrows and ncols
for i in range(len(model_evaluation_keys)): #length of 5
    if i<2:
        ax = axs[row, col]
        ax.set_title("{} vs epoch".format(model_evaluation_keys[i]))
        ax.plot(epoch_list, metric_list_train[i], color='#bcbd22', marker='o', mec = '#bcbd22', mfc = '#bcbd22', label = 'training {}'.format(model_evaluation_keys[i]))
        ax.plot(epoch_list, metric_list[i], color='#9467bd', marker='s', mec = '#9467bd', mfc = '#9467bd', label = 'test {}'.format(model_evaluation_keys[i]))
        ax.legend()
        ax.set_ylabel('{}'.format(model_evaluation_keys[i]))
        ax.set_xlabel('epochs')
        col += 1
        if col == 10:
            row += 1
            col = 0
    else:
        for j in range(len(training_df_cols)): #has a length of 28 or 31
            ax = axs[row, col]
            if i == 2:
                precision_dict_keys = list(precision_dict.keys())
                ax.set_title("{} {} vs epoch".format(precision_dict_keys[j], model_evaluation_keys[i]))
                ax.plot(epoch_list, precision_dict[precision_dict_keys[j]], color='#9467bd', marker='s', mec = '#9467bd', mfc = '#9467bd')
                ax.set_ylabel('precision')
                ax.set_xlabel('epochs')
            elif i == 3:
                recall_dict_keys = list(recall_dict.keys())
                ax.set_title("{} {} vs epoch".format(recall_dict_keys[j], model_evaluation_keys[i]))
                ax.plot(epoch_list, recall_dict[recall_dict_keys[j]], color='#9467bd', marker='s', mec = '#9467bd', mfc = '#9467bd')
                ax.set_ylabel('recall')
                ax.set_xlabel('epochs')
            elif i == 4:
                labels_dict_keys = list(labels_dict.keys())
                ax.set_title("{} accuracy vs epoch".format(labels_dict_keys[j]))
                ax.plot(epoch_list, labels_dict[labels_dict_keys[j]], color='#9467bd', marker='s', mec = '#9467bd', mfc = '#9467bd')
                ax.set_ylabel('accuracy')
                ax.set_xlabel('epochs')
            col += 1
            if col == 10:
                row += 1
                col = 0
            if row == 10:
                break  # Stop if we've filled all rows
        if row == 10:
            break
#plt.tight_layout()
plt.savefig('model_evaluation_plots/all_in_one.jpg', dpi=300)
plt.close()
print('all_in_one_succesfuly saved')

#confusion matrix plots at epochs 10, 20, 30, 40, 50 (28*5) or (31*5)
fig, axs = plt.subplots(31, 5, figsize=(35, 155), layout='constrained')
label_count = 2
row, col = 0, 0
data_FP = np.array([["TP", 'FP'], ['FN', 'TN']])
j_list = [int((epochs/5)-1), int(2*(epochs/5)-1), int(3*(epochs/5)-1), int(4*(epochs/5)-1), int(5*(epochs/5)-1)]
for i in range(len(training_df_cols)):
    for j in j_list:
        ax = axs[row, col]
        data = np.array([[TP[j][i].item(), FP[j][i].item()], [FN[j][i].item(), TN[j][i].item()]])
        im = ax.imshow(data, cmap='inferno', aspect='auto', vmin=0)
        ax.figure.colorbar(im, ax=ax)

        # Show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(label_count), labels=[training_df_cols[i], 'not {}'.format(training_df_cols[i])], rotation=45)
        ax.set_yticks(np.arange(label_count), labels=[training_df_cols[i], 'not {}'.format(training_df_cols[i])])
        ax.set_xlabel('True labels')
        ax.set_ylabel('Predicted labels')
        # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        # Loop over data dimensions and create text annotations.
        ax.set_title('Confusion matrix for {}'.format(training_df_cols[i]))
        texts = annotate_heatmap(im, valfmt="{x:.4g}")

        col += 1
        if col == 5:
            row += 1
            col = 0
        if row == 31:
            break  # Stop if we've filled all rows
    if row == 31:
        break
plt.savefig('model_evaluation_plots/class_confusion_matrix.jpg', dpi=300)
plt.close()
print('class_confusion_matrix succesfuly saved')

"""#full confusion matrix plotting 28x28
fig, axs = plt.subplots(1, 5, figsize=(50, 10), layout='constrained')
label_count = 28
col = 0
j_list = [9, 19, 29, 39, 49]
for j in j_list:
    ax = axs[col]
    data = np.array(FULL_CONFUSION[j])
    im = ax.imshow(data, cmap='inferno', aspect='auto', vmin=0)
    ax.figure.colorbar(im, ax=ax)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(label_count), labels=training_df_cols, rotation=45)
    ax.set_yticks(np.arange(label_count), labels=training_df_cols)
    ax.set_xlabel('True labels')
    ax.set_ylabel('Predicted labels')
    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    ax.set_title('Full confusion matrix 28x28 {}'.format(j))
    texts = annotate_heatmap_without_TP(im, valfmt="{x:.4g}")

    col += 1
plt.savefig('model_evaluation_plots/full_class_confusion_matrix.jpg', dpi=300)
plt.close()"""



print('model trained and saved')