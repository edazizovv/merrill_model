#


#
import pandas
import seaborn

import torch
from torch import nn


#


#
class Skeleton(nn.Module):
    """
    Supports Numerical Data Only!
    """

    def __init__(self, layers, layers_dimensions, layers_kwargs, activators, drops, verbose=-1, device=None):

        super().__init__()

        self.n = len(layers)
        self._verbose = verbose

        self._layers = layers
        self._layers_dimensions = layers_dimensions
        self._layers_kwargs = layers_kwargs
        self._activators = activators
        self._drops = drops

        self.layers = None
        self.epochs = None
        self.optimizer = None

        self._optimizer = None
        self._optimizer_kwargs = None

        self.input_shape = None
        self.train_losses = None
        self.validation_losses = None

        self.device = device

    def set_layers(self, input_shape):

        self.input_shape = input_shape

        layers = []
        for j in range(self.n):
            if self._layers[j].__name__ in ['Linear']:
                layers.append(self._layers[j](input_shape, self._layers_dimensions[j], **self._layers_kwargs[j]))
            elif self._layers[j].__name__ in ['RNN', 'GRU', 'LSTM']:
                layers.append(self._layers[j](input_shape, self._layers_dimensions[j], batch_first=True, **self._layers_kwargs[j]))
            if self._activators[j] is not None:
                layers.append(self._activators[j]())
            if self._drops[j] is not None:
                layers.append(nn.Dropout(self._drops[j]))
            input_shape = self._layers_dimensions[j]

        self.layers = nn.Sequential(*layers)
        if self.device is not None:
            self.layers.to(device=self.device)
        self.n = len(self.layers)

    def forward(self, x):

        x = torch.cat([x], 1)

        for j in range(self.n):
            if self.layers[j]._get_name() in ['RNN', 'GRU', 'LSTM']:
                x, _ = self.layers[j](x)
            else:
                x = self.layers[j](x)

        # x = self.layers(x)

        return x

    def fit(self, x_train, y_train, x_val, y_val, optimizer, optimizer_kwargs, loss_function, epochs=500):

        self.set_layers(input_shape=x_train.shape[-1])

        self._optimizer = optimizer
        self._optimizer_kwargs = optimizer_kwargs
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)

        if self._verbose != -1:
            print(self)

        self.epochs = epochs
        self.train_losses = []
        self.validation_losses = []

        for i in range(epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(x_train)
                    single_loss = loss_function(y_pred, y_train)
                else:
                    y_pred = self(x_val)
                    single_loss = loss_function(y_pred, y_val)

                self.optimizer.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.train_losses.append(train_lost)
                    single_loss.backward()
                    self.optimizer.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_losses.append(validation_lost)

            if self._verbose > 0:
                if (i % self._verbose) == 1:
                    print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))
        if self._verbose != -1:
            print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def predict(self, x):

        output = self(x)
        result = output.detach()

        if self.device.type == 'cuda':
            result = result.cpu()

        return result

    def plot(self):

        train = pandas.DataFrame(data={'epoch': range(self.epochs), 'loss': self.train_losses, 'on': ['train'] * self.epochs})
        validation = pandas.DataFrame(data={'epoch': range(self.epochs), 'loss': self.validation_losses, 'on': ['validation'] * self.epochs})
        result = pandas.concat((train, validation), axis=0, ignore_index=True)
        seaborn.lineplot(x='epoch', y='loss', hue='on', data=result, palette="tab10")
