import train
import data
import torch
import model

import platform
if platform.system() == 'Windows':
    import winsound

# setting ---------------------------------------------------
learning_rate = 1e-8
momentum = 0.9
num_epoch = 5
batch_size = 32
print_every = 20

device = torch.device('cuda')

if __name__ == '__main__':
    # network -------------------------------------------------------------------
    net = model.MyNet()
    net.load_state_dict(torch.load('model/model'))

    # dataset----------------------------------------------------------------------------------
    voc2012_train_dataset = data.Dataset('VOC2012/ImageSets/Main/train.txt', data_arg=True)
    voc2012_val_dataset = data.Dataset('VOC2012/ImageSets/Main/val.txt', data_arg=False)

    train_dataset = data.TensorDataset(voc2012_train_dataset)
    val_dataset = data.TensorDataset(voc2012_val_dataset)

    # training setup--------------------------------------------------------------------------------------
    optim = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    solv = train.Solver(
        model=net,
        optim=optim,
        train_data=train_dataset,
        val_data=val_dataset
    )

    train_losses, val_losses = solv.train(
        num_epoch=num_epoch,
        print_every=print_every,
        batch_size=batch_size,
        device=device
    )

    torch.save(net.state_dict(), 'model/model')

    for i in range(1, 10):
        winsound.Beep(i * 100, 200)
