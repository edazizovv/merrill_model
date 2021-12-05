
import torch
from torch import nn

import seaborn
import numpy
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot


class WrappedNumericOnlyGene:

    def __init__(self,
                 layers, layers_dimensions, layers_kwargs,
                 activators,
                 optimiser, optimiser_kwargs, loss_function,
                 processors,
                 interdrops,
                 epochs=500):

        self.optimiser = optimiser
        self.optimiser_kwargs = optimiser_kwargs
        self.loss_function = loss_function

        self.model = NumericOnlyGene(
                 layers=layers, layers_dimensions=layers_dimensions, layers_kwargs=layers_kwargs,
                 activators=activators,
                 processors=processors, interdrops=interdrops,
            optimiser=optimiser, optimiser_kwargs=optimiser_kwargs, # loss_function=loss_function,
                 epochs=epochs)

    def fit(self, X_train, y_train, X_val, y_val):

        # _optimizer = self.optimiser(self.model.parameters(), **self.optimiser_kwargs)
        self.model.fit(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, loss_function=self.loss_function)

    def plot_fit(self):
        self.model.plot_fit()

    def predict(self, X):
        return self.model.predict(X)

    def embed(self, X, n=-1):
        return self.model.embed(X, n=n)



class NumericOnlyGene(nn.Module):
    """
    Model architecture:

        -- Embedding
        -- EmbeddingDrop
        -- Preprocessor

        == LAYER X: -- Linear
                    -- Activator
                    -- Interprocessor (optional)
                    -- Interdrop      (optional)
    """

    def __init__(self,
                 layers, layers_dimensions, layers_kwargs,
                 activators,
                 processors, interdrops,
                 optimiser, optimiser_kwargs,
                 epochs
                 ):

        self.layers = None
        self.input_size = None
        self.optimiser = None
        self.batch_size = None
        # self.loss_function_ = loss_function
        self.aggregated_losses = []
        self.validation_losses = []

        self._layers = layers
        self._layers_dimensions = layers_dimensions
        self._layers_kwargs = layers_kwargs
        self._activators = activators
        self._processors = processors
        self._interdrops = interdrops
        self._optimiser = optimiser
        self._optimiser_kwargs = optimiser_kwargs
        self._epochs = epochs
        self._aggregated_losses = None
        self._validation_losses = None

        # self.hidden_cells = None

        #

        super().__init__()

    def initialise(self):

        all_layers = []

        input_size = self.input_size
        for j in range(len(self._layers_dimensions)):
            # if self._processors[j] is not None:
            #     all_layers.append(self._processors[j](self._layers_dimensions[j]))
            #     all_layers.append(self._processors[j](input_size))
            all_layers.append(self._layers[j](input_size, self._layers_dimensions[j], **self._layers_kwargs[j]))
            if self._activators[j] is not None:
                all_layers.append(self._activators[j]())
            if self._interdrops[j] is not None:
                all_layers.append(nn.Dropout(self._interdrops[j]))
            input_size = self._layers_dimensions[j]

        self.layers = nn.ModuleList(all_layers)

        #

    def _requires_hidden_cell(self, j):
        if self.layers is None:
            return self._layers[j].__name__ in ['RNN', 'GRU', 'LSTM']
        else:
            return self.layers[j]._get_name() in ['RNN', 'GRU', 'LSTM']

    def _is_technical(self, j):
        return self.layers[j]._get_name() in ['Dropout']

    def hidden_cell(self, j):
        # return torch.zeros(1, self.batch_size, self._layers_dimensions[j]).requires_grad_()
        return torch.zeros(1, self.batch_size, self.layers[j].hidden_size).requires_grad_()

    def forward(self, x):

        #

        self.batch_size = x.shape[0]

        #

        y = x

        for j in range(len(self.layers)):
            if not self._is_technical(j):
                if not self._requires_hidden_cell(j):
                    # if not self._requires_hidden_cell((j - 1)):
                    if y.dim() == 2:
                        y = self.layers[j](y)
                    elif y.dim() == 3:
                        y = self.layers[j](y[:, -1, :])
                    else:
                        raise Exception("Whoops...")
                else:
                    y, hidden_cell = self.layers[j](y, self.hidden_cell(j))
            else:
                y = self.layers[j](y)

        #

        return y

    def embed(self, x, n=-1):

        #

        self.batch_size = x.shape[0]

        #

        y = x

        for j in range((len(self.layers) + n)):
            if not self._is_technical(j):
                if not self._requires_hidden_cell(j):
                    # if not self._requires_hidden_cell((j - 1)):
                    if y.dim() == 2:
                        y = self.layers[j](y)
                    elif y.dim() == 3:
                        y = self.layers[j](y[:, -1, :])
                    else:
                        raise Exception("Whoops...")
                else:
                    y, hidden_cell = self.layers[j](y, self.hidden_cell(j))
            else:
                y = self.layers[j](y)

        #

        result = y.detach().numpy()
        return result

    def fit(self, X_train, y_train, X_val, y_val, loss_function):

        if self._requires_hidden_cell(0):
            self.input_size = X_train.shape[2]
        else:
            self.input_size = X_train.shape[1]

        self.initialise()
        self.optimiser = self._optimiser(self.parameters(), **self._optimiser_kwargs)

        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(self._epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(X_train)
                    single_loss = loss_function(y_pred, y_train)
                else:
                    y_pred = self(X_val)
                    single_loss = loss_function(y_pred, y_val)

                self.optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    # self.aggregated_losses.append(single_loss)
                    single_loss.backward()
                    self.optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    # self.validation_losses.append(single_loss)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def plot_fit(self):

        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.aggregated_losses, label='Train')
        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_losses, label='Validation')
        pyplot.legend(loc="upper left")
        pyplot.show()

    def predict(self, X_test):

        output = self(X_test)
        result = output.detach().numpy()

        return result


class _dwed_NumericOnlyGene(nn.Module):
    """
    Model architecture:

        -- Embedding
        -- EmbeddingDrop
        -- Preprocessor

        == LAYER X: -- Linear
                    -- Activator
                    -- Interprocessor (optional)
                    -- Interdrop      (optional)
    """

    def __init__(self,
                 layers, layers_dimensions,
                 activators,
                 processors, interdrops,
                 optimiser, optimiser_kwargs, # loss_function,
                 epochs,
                 batch_size=64
                 ):

        self.layers = None
        self.input_size = None
        self.optimiser = None
        self.batch_size = batch_size
        # self.loss_function_ = loss_function
        self.aggregated_losses = []
        self.validation_losses = []

        self._layers = layers
        self._layers_dimensions = layers_dimensions
        self._activators = activators
        self._processors = processors
        self._interdrops = interdrops
        self._optimiser = optimiser
        self._optimiser_kwargs = optimiser_kwargs
        self._epochs = epochs
        self._aggregated_losses = None
        self._validation_losses = None

        self.hidden_cells = None
        # self.parameters_list = None

        super().__init__()

    def initialise(self, numerical_shape):

        all_layers = []
        self.input_size = numerical_shape

        input_size = self.input_size
        for j in range(len(self._layers_dimensions)):
            # if self._processors[j] is not None:
            #     all_layers.append(self._processors[j](self._layers_dimensions[j]))
            #     all_layers.append(self._processors[j](input_size))
            all_layers.append(self._layers[j](input_size, self._layers_dimensions[j]))
            if self._activators[j] is not None:
                all_layers.append(self._activators[j]())
            if self._interdrops[j] is not None:
                all_layers.append(nn.Dropout(self._interdrops[j]))
            input_size = self._layers_dimensions[j]

        # self.layers = nn.Sequential(*all_layers)
        # self.layers = all_layers
        self.layers = nn.ModuleList(all_layers)

        # self.parameters_list = nn.ParameterList([x.parameters() for x in self.layers])

        self.hidden_cells = []  # .requires_grad_()
        for j in range(len(self.layers)):
            requires = self._requires_hidden_cell(j)
            if not requires:
                app = None
            else:
                app = torch.zeros(self.layers[j].input_size, 1, self.layers[j].hidden_size).requires_grad_()
            self.hidden_cells.append(app)

        # ?
        self.hidden_cells[0] = torch.zeros(1, 1, self.layers[0].hidden_size).requires_grad_()

    """
    def forward(self, x_numerical):

        if not str(self._layers[0]) == "<class 'torch.nn.modules.rnn.RNN'>":
            x = torch.cat([x_numerical], 1)
            x = self.layers(x)
        else:
            x = x_numerical.view(len(x_numerical), 1, -1)
            x, self.hidden_cell = self.layers(x, self.hidden_cell)
        return x
    """

    def _requires_hidden_cell(self, j):
        return self.layers[j]._get_name() in ['RNN']

    def forward(self, x):

        for j in range(len(self.layers)):

            if self.hidden_cells[j] is None:
                x = torch.cat([x], 1)
                x = self.layers[j](x)
            else:
                x = x.view(len(x), 1, -1)
                x, self.hidden_cells[j] = self.layers[j](x, self.hidden_cells[j])  # .detach())
                x = x[:, -1, :]

        return x

    def fit(self, X_train, y_train, X_val, y_val, loss_function):

        input_size = X_train.shape[1]

        self.initialise(numerical_shape=input_size)
        self.optimiser = self._optimiser(self.parameters(), **self._optimiser_kwargs)

        self.aggregated_losses = []
        self.validation_losses = []

        for i in range(self._epochs):
            i += 1
            for phase in ['train', 'validate']:

                if phase == 'train':
                    y_pred = self(X_train)
                    single_loss = loss_function(y_pred, y_train)
                else:
                    y_pred = self(X_val)
                    single_loss = loss_function(y_pred, y_val)

                self.optimiser.zero_grad()

                if phase == 'train':
                    train_lost = single_loss.item()
                    self.aggregated_losses.append(single_loss)
                    single_loss.backward()
                    self.optimiser.step()
                else:
                    validation_lost = single_loss.item()
                    self.validation_losses.append(single_loss)

            if i % 25 == 1:
                print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost,
                                                                                             validation_lost))
        print('epoch: {0:3} train loss: {1:10.8f} validation loss: {2:10.8f}'.format(i, train_lost, validation_lost))

    def plot_fit(self):

        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.aggregated_losses, label='Train')
        pyplot.plot(numpy.array(numpy.arange(self.epochs)), self.validation_losses, label='Validation')
        pyplot.legend(loc="upper left")
        pyplot.show()

    def predict(self, X_test):

        output = self(X_test)
        result = output.detach().numpy()

        return result
