import train
import data
import torch
import model
import winsound
import matplotlib.pyplot as plt

# setting ---------------------------------------------------
learning_rate = 1e-3
num_epoch = 20
batch_size = 18

device = torch.device('cuda')

if __name__ == '__main__':
    # network -------------------------------------------------------------------
    net = model.MyNet()
    net.load_state_dict(torch.load('model/model.pth'))

    # dataset----------------------------------------------------------------------------------
    voc2012_train_dataset = data.Dataset('VOC2007+2012/ImageSets/Main/train.txt', data_arg=True)
    voc2012_val_dataset = data.Dataset('VOC2007+2012/ImageSets/Main/val.txt', data_arg=True)

    train_dataset = data.TensorDataset(voc2012_train_dataset)
    val_dataset = data.TensorDataset(voc2012_val_dataset)

    # training setup--------------------------------------------------------------------------------------
    optim = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

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

    torch.save(net.state_dict(), 'model/model.pth')

    for i in range(1, 10):
        winsound.Beep(i * 100, 200)

    plt.plot(train_losses, 'b', val_losses, 'r--')
    plt.show()
