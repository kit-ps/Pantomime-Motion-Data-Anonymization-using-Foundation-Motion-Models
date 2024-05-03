import torch.nn as nn
import torch
import torch.nn.functional as F
from base import BaseModel


from .EKYT_networks import Classifier, SimpleDenseNet


class FacialMotionSimpleModel(BaseModel):
    def __init__(self, num_classes=116, frames=100, frame_size=7, num_hidden_layers=0, layer_size=100, return_feature=False):
        print(num_classes, frames, frame_size, num_hidden_layers, layer_size)
        super().__init__()
        self.fc_first = nn.Linear(frames * frame_size, layer_size)
        self.fc_last = nn.Linear(layer_size, num_classes)
        self.return_feature = return_feature

        self.fc_hidden = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.fc_hidden.append(nn.Linear(layer_size, layer_size))

    def forward(self, x):
        x = F.relu(self.fc_first(x))
        x = F.dropout(x, training=self.training)

        for i in range(len(self.fc_hidden)):
            x = F.relu(self.fc_hidden[i](x))
            x = F.dropout(x, training=self.training)

        x = self.fc_last(x)
        if self.return_feature:
            return x
        else:
            return F.log_softmax(x, dim=1)



class FacialMotionLSTMModel(BaseModel):
    def __init__(self, num_classes=116, frames=100, frame_size=7, layer_size=100, num_hidden_layers=1, return_feature=False):
        super().__init__()
        self.frames = frames
        self.frame_size = frame_size
        self.layer_size = layer_size
        self.num_hidden_layers = num_hidden_layers
        self.return_feature = return_feature

        if num_hidden_layers == 0:
            self.num_hidden_layers = 1

        self.lstm = nn.LSTM(self.frame_size, self.layer_size, self.num_hidden_layers, batch_first=True)
        self.fc = nn.Linear(self.layer_size, num_classes)


    def forward(self, x):
        hidden = (torch.randn(self.num_hidden_layers, x.size(0), self.layer_size).to(x.device),
                  torch.randn(self.num_hidden_layers, x.size(0), self.layer_size).to(x.device))


        x =  torch.permute(x, (0, 2, 1))
        out, hidden = self.lstm(x, hidden)

        # The out[:,-1,:] returns the last lstm output state for the entire sequence
        if self.return_feature:
            return out[:, -1, :]
        else:
            out = self.fc(out[:, -1, :])
            return F.log_softmax(out, dim=1)


class EyeKnowYouToo(BaseModel):
        def __init__(self, num_classes=116, frames=100, frame_size=7, layer_size=100, num_hidden_layers=1,
                         return_feature=False):
            super().__init__()

            self.embedder = SimpleDenseNet(depth=9, output_dim=128, initial_channels=frame_size)
            self.classifier = Classifier(self.embedder.output_dim, num_classes)
            self.return_feature = return_feature

        def forward(self, x):
            out = self.embedder(x)
            if self.return_feature:
                return out
            else:
                out = self.classifier(out)
                return F.log_softmax(out, dim=1)

'''
class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, return_feature=False):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.return_feature = return_feature

    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)

        out, (hn, cn) = self.lstm(x, (h0, c0))

        # The out[:,-1,:] returns the last lstm output state for the entire sequence
        if self.return_feature:
            return out[:, -1, :]
        else:
            out = self.fc(out[:, -1, :])
            return F.log_softmax(out, dim=1)



class FacialMotionLSTMFeatureExtractor(nn.Module):
    def __init__(self, frames=100, frame_size=7):
        super().__init__()
        self.frames = frames
        self.frame_size = frame_size
        self.hidden_dim = 100
        self.layer_dim = 1
        self.lstm = nn.LSTM(self.frame_size, self.hidden_dim, self.layer_dim)

    def forward(self, x):
        hidden = (torch.randn(1, 1, self.hidden_dim).to(x.device),
                  torch.randn(1, 1, self.hidden_dim).to(x.device))

        for i in range(self.frames):
            tmp = x[:,:,i].view(-1, 1, self.frame_size)
            out, hidden = self.lstm(tmp, hidden)

        out = out[:, -1, :]
        return out

class FacialMotionSimpleFeatureExtractor(BaseModel):
    def __init__(self, num_classes=10, frames=100, frame_size=7):
        super().__init__()
        self.fc1 = nn.Linear(frames * frame_size, 70)
        self.fc2 = nn.Linear(70, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x







class FacialMotionComplexModel(BaseModel):
    def __init__(self, num_classes=10, frames=100, frame_size=7):
        super().__init__()
        self.fc1 = nn.Linear(frames * frame_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class FacialMotionConvModel(BaseModel):
    def __init__(self, num_classes=10, frames=100, frame_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(7,10))
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=(7,1))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(91 * 10, 100)
        self.fc2 = nn.Linear(100, num_classes)

        self.frames = frames
        self.frame_size = frame_size


    def forward(self, x):
        x = x.view(-1, 1, self.frame_size, self.frames)
        x = F.relu(self.conv2_drop(self.conv1(x)))
        x = x.view(-1, 91 * 10)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
'''