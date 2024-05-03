import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class TripletTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (anchor, positive, negative, target) in enumerate(self.data_loader):
            anchor, positive, negative, target = (anchor.to(self.device), positive.to(self.device),
                                                  negative.to(self.device), target.to(self.device))

            self.optimizer.zero_grad()
            anchor_out = self.model(anchor)
            if self.criterion.__name__ == 'triplet_loss':
                positive_out = self.model(positive)
                negative_out = self.model(negative)
                loss = self.criterion(anchor_out, positive_out, negative_out, self.config["loss"]["args"])
            else:
                loss = self.criterion(anchor_out, target, self.config["loss"]["args"])
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            #for met in self.metric_ftns:
            #    self.train_metrics.update(met.__name__, met(anchor_out, positive_out, negative_out))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():

            all_train_outputs, all_train_targets = self.generate_embeddings(self.data_loader)
            average_per_label = self.average_per_label_func(all_train_outputs, all_train_targets)

            for batch_idx, (anchor, positive, negative, target) in enumerate(self.valid_data_loader):
                anchor, positive, negative, target = (anchor.to(self.device), positive.to(self.device),
                                                      negative.to(self.device), target.to(self.device))

                anchor_out = self.model(anchor)
                if self.criterion.__name__ == 'triplet_loss':
                    positive_out = self.model(positive)
                    negative_out = self.model(negative)
                    loss = self.criterion(anchor_out, positive_out, negative_out,  self.config["loss"]["args"])
                else:
                    loss = self.criterion(anchor_out, target, self.config["loss"]["args"])

                pred_label = self.predict_label(anchor_out, average_per_label)
                pred_label = torch.as_tensor(pred_label, device=self.device)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    if met.__name__ == "EER" or met.__name__ == "f1_score":
                        self.valid_metrics.update(met.__name__, met(pred_label, target, self.data_loader.dataset.get_num_classes()))
                    else:
                        self.valid_metrics.update(met.__name__, met(pred_label, target))
                #self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        # add histogram of model parameters to the tensorboard
        #for name, p in self.model.named_parameters():
        #    self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def predict_label(self, all_outputs, average_per_label):
        pred_labels = []

        for output in all_outputs:
            min = float("inf")
            min_label = 0
            for label in average_per_label:
                output = torch.reshape(output, (1, output.shape[-1]))
                average_per_label[label] = torch.reshape(average_per_label[label],
                                                         (1, average_per_label[label].shape[-1]))
                distance = torch.cdist(output, average_per_label[label])
                if distance < min:
                    min = distance
                    min_label = label

            pred_labels.append(min_label)
        return pred_labels

    def average_per_label_func(self, all_train_outputs, all_train_targets):
        average_per_label = {}
        for i in range(len(all_train_outputs)):
            label = all_train_targets[i]
            if label not in average_per_label:
                average_per_label[label] = [all_train_outputs[i]]
            else:
                average_per_label[label].append(all_train_outputs[i])
        for label in average_per_label:
            tmp = average_per_label[label][0]
            for i in range(1, len(average_per_label[label])):
                tmp += average_per_label[label][i]
            average_per_label[label] = tmp / len(average_per_label[label])
        return average_per_label

    def generate_embeddings(self, data_loader):
        all_outputs = []
        all_targets = []
        for i, (data, positive, negative, target) in enumerate(data_loader):
            data = data.to(self.device)
            output = self.model(data)
            for i in range(len(output)):
                all_outputs.append(output[i])
                all_targets.append(target[i].item())
        return all_outputs, all_targets
