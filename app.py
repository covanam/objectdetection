from PIL import Image, ImageFont, ImageDraw
import torch
import torchvision
import testmodel
import math
import data
table = (
        'bicycle',
        'bus',
        'car',
        'cat',
        'cow',
        'dog',
        'horse',
        'motorbike',
        'person',
        'sheep'
    )

totensor = torchvision.transforms.ToTensor()

net = testmodel.TestNet()
net.load_state_dict(torch.load('model/model'))
net.eval()

out = [c[2] for c in net(None)]

im = Image.open(r'C:\Users\Administrator\Downloads\objectdetection-master\VOC2012\JPEGImages\2008_000026.jpg')
im = im.resize((256, 256))
draw = ImageDraw.Draw(im)
fnt = ImageFont.truetype('Pillow/Tests/fonts/arial.ttf', 15)
print('\nv0')
for level in range(4):
    gg = 16 // 2**level - 1
    grid = 32 * 2**level
    for gx in range(gg):
        for gy in range(gg):
            for i in range(10):
                conf = out[level][i, gx, gy]
                if conf.item() < 0:
                    continue
                print(conf.item())

                x = out[level][10 + 4*i, gx, gy].item()
                y = out[level][11 + 4*i, gx, gy].item()
                w = out[level][12 + 4*i, gx, gy].item()
                h = out[level][13 + 4*i, gx, gy].item()
                x = (grid//2) * (gx + 1 + x)
                y = (grid//2) * (gy + 1 + y)
                w = grid * math.exp(w)
                h = grid * math.exp(h)
                print(x, y, w, h)
                tag = table[i]

                draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), outline='red')
                draw.text((x-w/2, y-h/2), tag, font=fnt, fill=(255, 0, 0, 128))

im.save('hi.png')


