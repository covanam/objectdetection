import torch

def Loss(i
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
        train_dataloader = torch.utils.data.DataLoader(self.train_data, batch_size, shuffle=True, num_workers=0)
        if self.val_data is not None:
            val_dataloader = torch.utils.data.DataLoader(self.val_data, 10, shuffle=True, num_workers=0)

        for e in range(num_epoch):
            print('epoch', e, 'begin training---------------------------')
            self.model.train()

            train_loss = MeanLoss()
            for x, true_tensor in train_dataloader:
                x = x.to(device)
                true_tensor = true_tensor.to(device)

                self.optim.zero_grad()

                # forward path
                pred = self.model(x)
                loss = self.loss_fn(pred, true_tensor)

                # print out the accuracy
                if count % print_every == 0:
                    self.check_acc(loss.item(), pred, true_tensor)

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

                self.check_acc(loss, pred, true_tensor)

                val_loss.add(loss.item())
            val_loss = val_loss.calc()
            print('val loss:', val_loss)

        self.model = self.model.to(torch.device('cpu'))

    @staticmethod
    def check_acc(loss, pred_out, true_out):
        sampled_pred_out = torch.argmax(pred_out, dim=1)
        true_map = sampled_pred_out == true_out

        ground_mask = (true_out == 0)
        obj_mask = (true_out != 0) - (true_out == -1)

        num_gnd = torch.sum(ground_mask).item()
        num_obj = torch.sum(obj_mask).item()

        if num_gnd:
            ground_acc = torch.sum(true_map[ground_mask]).item() / num_gnd
        else:
            ground_acc = -1
        if num_obj:
            obj_acc = torch.sum(true_map[obj_mask]).item() / num_obj
        else:
            obj_acc = -1
        
        print('\tgnd:{:4.2f}  obj:{:4.2f}  loss:{:7.5f}'.format(ground_acc, obj_acc, loss))



