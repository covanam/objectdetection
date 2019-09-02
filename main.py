import train
import data
import torch
import mobilenetv3
import winsound

# setting ---------------------------------------------------
learning_rate = 1e-4
momentum = 0
num_epoch = 1
batch_size = 10

device = torch.device('cuda')

if __name__ == '__main__':
    # network -------------------------------------------------------------------
    net = mobilenetv3.MyNet()
    net.load_state_dict(torch.load('model/model'))

    # dataset----------------------------------------------------------------------------------
    voc2012_train_dataset = data.Dataset('VOC2012/ImageSets/Main/train.txt', data_arg=True)
    voc2012_val_dataset = data.Dataset('VOC2012/ImageSets/Main/val.txt', data_arg=True)

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
        batch_size=batch_size,
        device=device
    )

    torch.save(net.state_dict(), 'model/model')

    for i in range(1, 10):
        winsound.Beep(i * 100, 200)
