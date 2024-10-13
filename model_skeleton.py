
import torch
from torch import nn


#convolutions -> relu -> pooling -> flatten -> fully connected -> softmax -> labels
#might be helpful https://www.youtube.com/watch?v=pDdP0TFzsoQ
#ARCHITECTURE OF THE CUSTOM CNN
class cNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #convolution layer
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),  # 3 in channels as we are using jpg (i.e. rgb)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        #Calculate the size after convolutions and pooling
        self.fc_input_size = self._get_conv_output((3, 400, 400))

        #fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 31)
        )

    #the below lets us calculate the shape of the tensor that should be passed to the linear layer
    def _get_conv_output(self, shape):
        input = torch.rand(1, *shape) #the * here unpacks shape which is a tuple
        output = self.features(input)
        return int(torch.prod(torch.tensor(output.size())))


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) #this now does our flattening although I am not sure whether view is getting its arguments in the correct order
        #x = x.view(-1, x.size(0))
        logits = self.classifier(x)
        return logits


