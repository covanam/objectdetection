import torch
import torch.nn.functional as F

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

        obj_loss = self._calc_obj_loss(x, target)
        bbox_loss = self._calc_bbox_loss(x, target)

        loss = obj_loss + bbox_loss

        return loss

    def _calc_obj_loss(self, x, target):
        obj_loss = torch.zeros(1, dtype=torch.float, device=self.device)
        for level in range(4):
            pos = x[level][:, 0:10][target[level][:, 0:10] == 1]
            neg = x[level][:, 0:10][target[level][:, 0:10] == 0]

            num_p = pos.shape[0]
            num_n = neg.shape[0]
            num_n_train = min(max(3 * num_p, 1), num_n)

            obj_loss += F.binary_cross_entropy_with_logits(
                pos,
                torch.ones_like(pos),
                reduction='sum'
            )

            neg = neg.topk(k=num_n_train)[0]

            obj_loss += F.binary_cross_entropy_with_logits(
                neg,
                torch.zeros_like(neg),
                reduction='sum'
            ) * max(1, num_p / num_n_train)  # balanced

        return obj_loss

    def _calc_bbox_loss(self, x, target):
        bbox_loss = torch.zeros(1, dtype=torch.float, device=self.device)
        for level in range(4):
            mask = (target[level][:, 14] != 0).unsqueeze(1).expand_as(x[level][:, 10:14])
            bbox_loss += F.mse_loss(
                x[level][:, 10:14][mask],
                target[level][:, 10:14][mask],
                reduction='sum'
            )

        return bbox_loss


class MeanLoss:
    def __init__(self):
        self.count = 0
        self.total_loss = 0

    def add(self, loss):
        self.count += 1
        self.total_loss += loss

    def calc(self):
        mean_loss = self.total_loss / self.count
        self.count = 0
        self.total_loss = 0
        return mean_loss


class Solver:
    def __init__(self, model, optim, loss_fn, train_data, val_data=None):
        self.train_data = train_data
        self.val_data = val_data
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

    def train(self, num_epoch=10, print_every=5, batch_size=10, device=torch.device('cpu')):
        self.model = self.model.to(device)

        count = 0
        train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size, shuffle=False, num_workers=0)
        if self.val_data is not None:
            val_dataloader = torch.utils.data.DataLoader(self.val_data, 10, shuffle=True, num_workers=0)

        for e in range(num_epoch):
            print('epoch', e, 'begin training---------------------------')
            self.model.train()

            train_loss = MeanLoss()
            for x, target in train_dataloader:
                x = x.to(device)
                #for c in true_tensor:
                #    for cc in c:
                #        cc = cc.to(device)

                self.optim.zero_grad()

                # forward path
                pred = self.model(x)
                loss = self.loss_fn(pred, target)

                # print out the accuracy
                if count % print_every == 0:
                    print('\t', loss.item())

                loss.backward()
                self.optim.step()

                count += 1
                train_loss.add(loss.item())
            train_loss = train_loss.calc()
            print('train_loss:', train_loss)

            if self.val_data is None:
                continue
            print('evaluate:')
            self.model.eval()

            val_loss = MeanLoss()
            for x, true_tensor in val_dataloader:
                x = x.to(device)
                true_tensor = true_tensor.to(device)

                # forward path
                pred = self.model(x)
                loss = self.loss_fn(pred, true_tensor)

                loss.backward()
                self.optim.zero_grad()  # no param update here!
                self.optim.step()  # to free up space,

                print('/t', loss.item())

                val_loss.add(loss.item())
            val_loss = val_loss.calc()
            print('val loss:', val_loss)

        self.model = self.model.to(torch.device('cpu'))
