import visdom
import torch
import numpy as np

class Visualizer(object):
    def __init__(self, vis_flag=True):
        self.vis_flag = vis_flag
        if self.vis_flag:
            self.loss_vis = visdom.Visdom(port=8097)
            self.loss_win = self.iteration_loss()

    def iteration_loss(self):
        epoch_loss = self.loss_vis.line(
            X=torch.zeros((1,)),
            Y=torch.zeros((1,)),
            opts=dict(
                xlabel='epoch',
                ylabel='Loss',
                title='Training loss',
                legend=['Loss'])
        )
        return epoch_loss

    def append_loss(self, iteration, loss_item):
        if self.vis_flag:
            self.loss_vis.line(
                X=torch.ones((1,)) * iteration,
                Y=torch.ones((1,)) * loss_item,
                win=self.loss_win,
                update='append')
