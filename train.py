import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class_tags = {'bicycle': 0,
              'bus': 1,
              'car': 2,
              'cat': 3,
              'cow': 4,
              'dog': 5,
              'horse': 6,
              'motorbike': 7,
              'person': 8,
              'sheep': 9}


class LossFunction:
    def __init__(self):
        pass

    def __call__(self, x, target):
        self.device = x[0].device
        batchsize = x[0].shape[0]

        obj_loss = self._calc_obj_loss(x, target)
        bbox_loss = self._calc_bbox_loss(x, target)

        loss = obj_loss + 0.5 * bbox_loss

        return loss / batchsize

    def _calc_obj_loss(self, x, target):
        obj_loss = torch.zeros(1, dtype=torch.float, device=self.device)  # obj_loss = 0

        for level in range(4):
            for i in range(10):  # we deal with 10 classes seperately
                pos = x[level][:, i][target[level][:, i] == 1]  # output that will be trained to be positive
                neg = x[level][:, i][target[level][:, i] == 0]  # output that will be trained to be negative

                if pos.shape[0] == 0:
                    continue  # nothing to train here

                # loss for positive output:
                obj_loss += F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos), reduction='sum')

                # for each positive output, pick 3 negative output for training
                num_p, num_n = pos.shape[0], neg.shape[0]
                num_n_train = min(3 * num_p, num_n)  # must not exceed num_n for obvious reason
                neg = neg.topk(k=num_n_train)[0]  # pick highest

                # loss for negative output
                obj_loss += F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg), reduction='sum') * (1/3)

        return obj_loss

    def _calc_bbox_loss(self, x, target):
        bbox_loss = torch.zeros(1, dtype=torch.float, device=self.device)
        for level in range(4):
            for i in range(10):
                mask = (target[level][:, i:i+1] == True).expand((-1, 4, -1, -1))

                if mask.sum().item() == 0:
                    continue  # nothing to train here
                
                bbox_loss += F.mse_loss(
                    x[level][:, 10+4*i:14+4*i][mask],
                    target[level][:, 10+4*i:14+4*i][mask],
                    reduction='sum'
                )

        return bbox_loss


class LossMonitor:
    def __init__(self):
        self.batch_count = 0
        self.accum_loss = 0
        self.loss_history = []

    def add(self, loss):
        self.batch_count += 1
        self.accum_loss += loss

    def __enter__(self):
        return self

    def __exit__(self, a, b, c):
        avg_loss = self.accum_loss / self.batch_count
        self.batch_count = 0
        self.accum_loss = 0
        self.loss_history.append(avg_loss)
        print('\tavg:', avg_loss)


class Solver:
    def __init__(self, model, optim, loss_fn=LossFunction(), train_data=None, val_data=None):
        self.train_data = train_data
        self.val_data = val_data
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.device = torch.device('cpu')

    def train(self, num_epoch=10, batch_size=10, device=torch.device('cpu')):
        # push model to GPU if needed
        self.device = device
        self.model = self.model.to(device)

        # loss monitors
        train_loss_monitor = LossMonitor()
        val_loss_monitor = LossMonitor()

        # data loaders
        pin_memory = device.type == 'cuda'
        print(pin_memory)
        train_dataloader = torch.utils.data.DataLoader(
            self.train_data, batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=pin_memory
        )
        if self.val_data is not None:
            val_dataloader = torch.utils.data.DataLoader(
                self.val_data, 20, shuffle=True, num_workers=4, pin_memory=pin_memory
            )

        # main part
        for e in range(num_epoch):
            print('epoch', e, '--------------------------------')
            # train ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            self.model.train()
            with train_loss_monitor as monitor:
                for x, target in train_dataloader:
                    loss = self._train_step(x, target)
                    monitor.add(loss)
                    print(loss)

            # if there is validation data, then validate ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if self.val_data is None:
                continue

            self.model.eval()
            with val_loss_monitor as monitor:
                for x, target in val_dataloader:
                    loss = self._val_step(x, target)
                    monitor.add(loss)

        return train_loss_monitor.loss_history, val_loss_monitor.loss_history

    def _train_step(self, x, target):
        # move data to gpu if needed
        x = x.to(self.device)
        target = [t.to(self.device) for t in target]

        self.optim.zero_grad()

        # forward path
        pred = self.model(x)
        loss = self.loss_fn(pred, target)

        # backward path
        loss.backward()
        self.optim.step()

        return loss.item()

    def _val_step(self, x, target):
        with torch.no_grad():
            x = x.to(self.device)
            target = [t.to(self.device) for t in target]

            # forward path
            pred = self.model(x)
            loss = self.loss_fn(pred, target)

        return loss.item()


