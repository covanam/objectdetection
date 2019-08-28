import train
import data
import torch
import model

# setting ---------------------------------------------------
learning_rate = 1
momentum = 0
num_epoch = 1
batch_size = 2
print_every = 1

device = torch.device('cpu')


# network -------------------------------------------------------------------
net = model.model()#mnasnet.MNASNet(1.0))
#net.load_state_dict(torch.load('model/model'))
# dataset----------------------------------------------------------------------------------
dataset = data.Dataset('VOC2012/ImageSets/Main/train.txt', data_arg=True, size=(256, 256))
val_dataset = data.Dataset('VOC2012/ImageSets/Main/val.txt', data_arg=False, size=(256, 256))

# training --------------------------------------------------------------------------------------
optim = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

loss_fn = train.LossFunction()

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
