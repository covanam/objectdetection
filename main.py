import train
import data
import torch
import model
import winsound
# setting ---------------------------------------------------
learning_rate = 1e-8
momentum = 0.9
num_epoch = 5
batch_size = 32
print_every = 20

device = torch.device('cuda')

if __name__ == '__main__':
    with open('trainloss.txt', 'a') as f:
        f.write('rate:' + str(learning_rate) + '\n')
    with open('valloss.txt', 'a') as f:
        f.write('rate:' + str(learning_rate) + '\n')
    # network -------------------------------------------------------------------
    net = model.MyNet()
    net.load_state_dict(torch.load('model/model'))

    # dataset----------------------------------------------------------------------------------
    dataset = data.Dataset('VOC2012+2007/ImageSets/Main/train.txt', data_arg=True, size=(256, 256))
    val_dataset = data.Dataset('VOC2012+2007/ImageSets/Main/val.txt', data_arg=False, size=(256, 256))

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

    for i in range(1, 10):
        winsound.Beep(i* 100, 200)
