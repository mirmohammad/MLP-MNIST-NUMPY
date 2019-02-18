import gzip
import pickle

import numpy as np
from tqdm import tqdm


class MnistDataset(object):
    def __init__(self, path_to_gz):
        f = gzip.open(path_to_gz, 'rb')
        (self.train_images, self.train_labels), \
        (self.valid_images, self.valid_labels), \
        (self.test_images, self.test_labels) = pickle.load(f, encoding='latin1')
        self.num_train = self.train_images.shape[0]
        self.num_valid = self.valid_images.shape[0]
        self.num_test = self.test_images.shape[0]


class Linear(object):
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        self.weights = np.zeros((in_features, out_features))
        self.bias = np.zeros((1, out_features))

    def reset_params(self, mode='zero'):
        if mode == 'zero':
            self.weights = np.zeros((self.in_features, self.out_features))
        elif mode == 'normal':
            self.weights = np.random.normal(0, 1, (self.in_features, self.out_features))
        elif mode == 'glorot':
            dl = np.sqrt(6.0 / (self.in_features + self.out_features))
            self.weights = np.random.uniform(-dl, dl, (self.in_features, self.out_features))

    def forward(self, x):
        self.x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, dy):
        self.dw = np.dot(self.x.T, dy)
        self.db = np.sum(dy, axis=0, keepdims=True)
        return np.dot(dy, self.weights.T)

    def update(self, lr, reg):
        self.dw = self.dw + reg * self.weights
        self.weights = self.weights - lr * self.dw
        self.bias = self.bias - lr * self.db


class ReLU(object):
    def __init__(self):
        pass

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dy):
        dy[self.x <= 0] = 0
        return dy
        # x = self.x
        # x[x <= 0] = 0
        # x[x > 0] = 1
        # return np.dot(dy, x)


class CrossEntropyLoss(object):
    def __init__(self):
        self.epsilon = 1e-100

    def forward(self, x, target):
        bs = x.shape[0]
        max_out = x - np.amax(x, axis=1).reshape(bs, 1)
        exp_max_out = np.exp(max_out)
        probs = (exp_max_out + self.epsilon) / np.sum(exp_max_out, axis=1, keepdims=True)
        ll = -np.log(probs[range(bs), target])
        return np.sum(ll) / bs, probs

    def backward(self, probs, target):
        bs = probs.shape[0]
        dy = probs
        dy[range(bs), target] -= 1
        return dy / bs


class NN(object):
    def __init__(self, in_features=784, out_features=10, hidden_dims=(512, 512), lr=0.001, reg=0.):
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.reg = reg
        self.hl_1 = Linear(in_features, hidden_dims[0])
        self.r1 = ReLU()
        self.hl_2 = Linear(hidden_dims[0], hidden_dims[1])
        self.r2 = ReLU()
        self.ol = Linear(hidden_dims[1], out_features)

    def init_params(self, mode='zero'):
        self.hl_1.reset_params(mode)
        self.hl_2.reset_params(mode)
        self.ol.reset_params(mode)

    def forward(self, x):
        xx = self.hl_1.forward(x)
        xx = self.r1.forward(xx)
        xx = self.hl_2.forward(xx)
        xx = self.r2.forward(xx)
        xx = self.ol.forward(xx)
        return xx

    def backward(self, dy):
        dy = self.ol.backward(dy)
        dy = self.r2.backward(dy)
        dy = self.hl_2.backward(dy)
        dy = self.r1.backward(dy)
        dy = self.hl_1.backward(dy)
        return dy

    def update(self):
        self.ol.update(self.lr, self.reg)
        self.hl_2.update(self.lr, self.reg)
        self.hl_1.update(self.lr, self.reg)


# Configuration

dataset = MnistDataset('/Users/mir/PycharmProjects/IFT6135_LAB1/mnist.pkl.gz')

np.random.seed(42)

epochs = 100
batch_size = 1000

image_res = 28 * 28
classes = 10
learning_rate = 1e-2
reg_factor = 1e-5

hiddens = (512, 512)
init_mode = ['zeros', 'normal', 'glorot'][2]

# Init model

model = NN(in_features=image_res, out_features=classes, hidden_dims=hiddens, lr=learning_rate, reg=reg_factor)
criterion = CrossEntropyLoss()
model.init_params(mode=init_mode)


def train(ep):
    num_images = 0
    running_loss = 0.
    running_acc = 0.
    num_splits = dataset.num_train // batch_size
    train_loader = zip(np.split(dataset.train_images, num_splits), np.split(dataset.train_labels, num_splits))
    monitor = tqdm(train_loader, desc='Train')
    for images, labels in monitor:
        out = model.forward(images)
        loss, probs = criterion.forward(out, labels)
        preds = np.argmax(probs, axis=1)

        num_images += images.shape[0]
        running_loss += loss * images.shape[0]
        running_acc += (preds == labels).sum()

        model.backward(criterion.backward(probs, labels))
        model.update()

        monitor.set_postfix(epoch=ep, loss=running_loss / num_images, accuracy=running_acc / num_images)

    epoch_loss = running_loss / num_images
    epoch_acc = running_acc / num_images

    return epoch_loss, epoch_acc


def valid(ep):
    num_images = 0
    running_loss = 0.
    running_acc = 0.
    num_splits = dataset.num_valid // batch_size
    valid_loader = zip(np.split(dataset.valid_images, num_splits), np.split(dataset.valid_labels, num_splits))
    monitor = tqdm(valid_loader, desc='Validation')
    for images, labels in monitor:
        out = model.forward(images)
        loss, probs = criterion.forward(out, labels)
        preds = np.argmax(probs, axis=1)

        num_images += images.shape[0]
        running_loss += loss * images.shape[0]
        running_acc += (preds == labels).sum()

        # model.backward(criterion.backward(probs, labels))
        # model.update()

        monitor.set_postfix(epoch=ep, loss=running_loss / num_images, accuracy=running_acc / num_images)

    epoch_loss = running_loss / num_images
    epoch_acc = running_acc / num_images

    return epoch_loss, epoch_acc


def test(ep):
    num_images = 0
    running_loss = 0.
    running_acc = 0.
    num_splits = dataset.num_test // batch_size
    test_loader = zip(np.split(dataset.test_images, num_splits), np.split(dataset.test_labels, num_splits))
    monitor = tqdm(test_loader, desc='Test')
    for images, labels in monitor:
        out = model.forward(images)
        loss, probs = criterion.forward(out, labels)
        preds = np.argmax(probs, axis=1)

        num_images += images.shape[0]
        running_loss += loss * images.shape[0]
        running_acc += (preds == labels).sum()

        # model.backward(criterion.backward(probs, labels))
        # model.update()

        monitor.set_postfix(epoch=ep, loss=running_loss / num_images, accuracy=running_acc / num_images)

    epoch_loss = running_loss / num_images
    epoch_acc = running_acc / num_images

    return epoch_loss, epoch_acc


def log_results(file_name, losses, accuracies):
    file = open(file_name, 'w+')
    for x, y in zip(losses, accuracies):
        file.write('{:.3f},'.format(x))
        file.write('{:.3f}'.format(y))
        file.write('\n')
    file.close()


if __name__ == '__main__':

    train_losses = []
    train_accs = []

    valid_losses = []
    valid_accs = []

    test_losses = []
    test_accs = []

    for epoch in range(epochs):
        train_loss, train_acc = train(epoch + 1)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        valid_loss, valid_acc = valid(epoch + 1)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        test_loss, test_acc = test(epoch + 1)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    log_results('train_loss_' + init_mode + '.csv', train_losses, train_accs)
    log_results('valid_loss_' + init_mode + '.csv', valid_losses, valid_accs)
    log_results('test_loss_' + init_mode + '.csv', test_losses, test_accs)
