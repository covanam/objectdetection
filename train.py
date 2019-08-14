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
    def __init__(self, obj_weight, bbox_weight, class_weight, pos_obj_weight):
        self.positive_weight = positive_weight
        self.object_weight = object_weight
        self.bndbox_weight = bndbox_weight
        self.classify_weight = classify_weight
        self.pos_obj_weight = pos_obj_weight
    def __call__(self, input, object_holders):
        logistic_loss = F.binary_cross_entropy_with_logits
        
        device = input.device
        batchsize = input[0][0]
        
        # 
        self.target_obj = []
        self.target_bbox = []
        self.target_class = []
        self.obj_mask = []
        
        for i in range(4):
            self.target_obj.append(torch.zeros((batchsize, 15, 15), dtype=torch.float, device=device))
            self.obj_mask.append(torch.zeros((batchsize, 15, 15), dype=torch.uint8, device=device))
            self.target_bbox.append(torch.empty((batchsize, 15, 15, 4), dtype=torch.float, device=device))
            self.target_class.append(torch.empty((batchsize, 15, 15), dtype=torch.long, device=device))
        
        objects_list = [holder.data for holder in object_holders]
        
        for idx, objects in enumerate(objects_list):
            true_out[idx] = self._construct_out(objects)
        
        obj_loss = []
        bndbox_loss = []
        class_loss = []
        for i in range(4):
            objectness_loss.append(
                logistic_loss(input, target_obj_1, reduction='sum', pos_weight=self.pos_obj_weight[i]) / self.pos_obj_weight[i]
            )
            boundbox_loss.append(
                F.mse_loss(input, target_bbox_1, reduction='sum')
            )
            classify_loss_1.append(
                F.cross_entropy(input, target, weight=None,ignore_index=-1, reduction='sum')
            )
        
        obj_loss = sum(obj_loss)
        bndbox_loss = sum(bndbox_loss)
        class_loss = sum(class_loss)
        
        loss self.obj_weight * obj_loss + self.bbox_weight * bndbox_loss, self.class_weight * class_loss
        
        return loss
        
    
    def _construct_target_tensor(objects, idx):
        for obj in objects:
            level = self._assign_level(obj)
            grid_x, grid_y, rx, ry = self._assign_grid(obj, level)
            
            x = obj.x()
            
            self.target_obj[level][idx, grid_x, grid_y] = 1
            self.target_class[level][idx, grid_x, grid_y] = self._encode_class(obj.tag)
            self.target_bbox[level][idx, grid_x, grid_y, 0] = rx
            self.target_bbox[level][idx, grid_x, grid_y, 1] = ry
            self.target_bbox[level][idx, grid_x, grid_y, 2] = math.log(obj.w())
            self.target_bbox[level][idx, grid_x, grid_y, 3] = math.log(obj.h())
            
            
    
    @staticmethod
    def _assign_level(obj):
        area = obj.area()
        if area < 2048:  # 0 -> 2x32x32
            return 0
        if area < 8192:  # 2x32x32 -> 2x64x64
            return 1
        if area < 32768:  # 2x64x64 -> 2x128x128
            return 2
        return 3  # 2x128x128 -> ...
    
    @staticmethod
    def _assign_grid(obj, level):
        if level == 3:
            return 0, 0
        
        grid_size = 32 * 2**level
        grid = 16 // (2**level) - 1  # 15, 7, 3, 1
        micro_grid = 32 // (level + 1)  # 32 16 8 4
        
        
        grid_x = obj.x() // (256 // micro_grid)
        grid_x = (grid_x + 1) // 2 - 1
        if grid_x < 0:
            grid_x = 0
        if grid_x > grid - 1:
            grid_x = grid - 1
            
        grid_y = obj.y() // (256 // micro_grid)
        grid_y = (grid_y + 1) // 2 - 1
        if grid_y < 0:
            grid_y = 0
        if grid_y > grid - 1:
            grid_y = grid - 1
        
        relative_x = obj.x() / (grid_size // 2) - (grid_x + 1)
        relative_y = obj.y() / (grid_size // 2) - (grid_y + 1)
        
        return grid_x, grid_y, relative_x, relative_y
    
    @staticmethod
    def _encode_class(tag):
        return class_tags[tag]
        
        
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



