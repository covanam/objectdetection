import train
import torchvision
import data
import torch
import model

# setting ---------------------------------------------------
learning_rate = 1e-4
momentum = 0.9
num_epoch = 1
batch_size = 50
print_every = 10

device = torch.device('cuda')
ground_weight = 0.1


# network -------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Conv2d(512, 11, 1, padding=0, stride=1, bias=True) 
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

origin_net = torchvision.models.vgg16(False)

net = Net()
net.features = origin_net.features

net.load_state_dict(torch.load('model/model'))

net = net.to(device)

del origin_net


#net = model.MyNetwork()
#net.load_state_dict(torch.load('model/model'))



# dataset----------------------------------------------------------------------------------
dataset = data.Dataset('VOC2012/ImageSets/Main/train.txt', data_arg=True, size=(224, 224))
val_dataset = data.Dataset('VOC2012/ImageSets/Main/val.txt', data_arg=False, size=(224, 224))


# training --------------------------------------------------------------------------------------
optim = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

weight = torch.tensor((ground_weight, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(
    weight=weight,
    ignore_index=-1,
    reduction='mean'
)

solv = train.Solver(
    model=net,
    optim=optim,
    loss_fn=loss_fn,
    train_data=dataset,
    val_data=val_dataset
)

solv.train(num_epoch=num_epoch, print_every=print_every, batch_size=batch_size, device=device)

del data
del solv
del optim

torch.save(net.state_dict(), 'model/model')
