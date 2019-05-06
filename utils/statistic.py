import pdb
import numpy as np
import matplotlib.pyplot as plt


class Stat(object):
    def __init__(self, num_classes):
        self.prec1_mean = []
        self.prec5_mean = []
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def reset_statistic(self):
        self.prec1_mean = []
        self.prec5_mean = []
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update_accuracy(self, output, target, topk):
        prec1, prec5 = self.accuracy(output.data.cpu(), target.cpu(), topk=topk)
        self.prec1_mean.append(prec1)
        self.prec5_mean.append(prec5)

    def show_accuracy(self, save_path):
        self.plot_confusion_matrix(save_path=save_path)
        return np.mean(self.prec1_mean), np.mean(self.prec5_mean)

    def accuracy(self, output, target, topk=(1,)):
        max_k = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(max_k, 1, True, True)
        for idx, tar in enumerate(target):
            self.confusion_matrix[tar, pred[idx, 0]] += 1
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def plot_confusion_matrix(self, save_path=None, normalize=True, title='Confusion matrix',
                              cmap=plt.cm.jet):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        np.save(save_path + ".npy", self.confusion_matrix)
        if normalize:
            cm = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig(save_path + ".jpg")
        plt.close()
