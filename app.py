from PIL import Image, ImageFont, ImageDraw
import torch
import torchvision
import model
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

net = model.model()
#net.load_state_dict(torch.load('model/model'))
net.eval()

im = Image.open('VOC2012/JPEGImages/2008_000036.jpg').resize((256, 256))
x = torchvision.transform.ToTensor()(im)

out1, out2, out3, out4 = [c[0] for c in net(x)]

draw = ImageDraw.Draw(im)
fnt = ImageFont.truetype('Pillow/Tests/fonts/arial.ttf', 15)
print('\nv0')
for gx in range(15):
    for gy in range(15):
        conf = out1[0:10, gx, gy].max()
        if conf.item() < 0.1:
            continue
        cid = out1[0:10, gx, gy].argmax()
        x = out1[10, gx, gy]
        y = out1[11, gx, gy]
        w = out1[12, gx, gy]
        h = out1[13, gx, gy]
        x = 16 * (gx + 1) + 16 * x
        y = 16 * (gy + 1) + 16 * y
        w = 32 * math.exp(w)
        h = 32 * math.exp(h)
        tag = table[cid.item()]
        print(tag)
        draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), outline='red')
        draw.text((x-w/2, y-h/2), 'lv1'+tag, font=fnt, fill=(255, 0, 0, 128))

print('\nv1')
for gx in range(7):
    for gy in range(7):
        conf = out2[0:10, gx, gy].max()
        if conf.item() < 0.1:
            continue
        cid = out2[0:10, gx, gy].argmax()
        x = out2[10, gx, gy]
        y = out2[11, gx, gy]
        w = out2[12, gx, gy]
        h = out2[13, gx, gy]
        x = 32 * (gx + 1) + 32 * x
        y = 32 * (gy + 1) + 32 * y
        w = 64 * math.exp(w)
        h = 64 * math.exp(h)
        tag = table[cid.item()]
        print(tag)
        draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), outline='red')
        draw.text((x-w/2, y-h/2), 'lv2'+tag, font=fnt, fill=(255, 0, 0, 128))
print('\nv2')
for gx in range(3):
    for gy in range(3):
        conf = out3[0:10, gx, gy].max()
        if conf.item() < 0.1:
            continue

        cid = out3[0:10, gx, gy].argmax()
        x = out3[10, gx, gy].item()
        y = out3[11, gx, gy].item()
        w = out3[12, gx, gy].item()
        h = out3[13, gx, gy].item()
        x = 64 * (gx + 1) + 64 * x
        y = 64 * (gy + 1) + 64 * y
        w = 128 * math.exp(w)
        h = 128 * math.exp(h)
        tag = table[cid.item()]
        print(tag)
        draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), outline='red')
        draw.text((x-w/2, y-h/2), 'lv3'+tag, font=fnt, fill=(255, 0, 0, 128))

print('\nv3')
for gx in range(1):
    for gy in range(1):
        conf = out4[0:10, gx, gy].max()
        if conf.item() < 0.1:
            continue

        cid = out4[0:10, gx, gy].argmax()
        x = out4[10, gx, gy]
        y = out4[11, gx, gy]
        w = out4[12, gx, gy]
        h = out4[13, gx, gy]
        x = 128 * (gx + 1) + 128 * x
        y = 128 * (gy + 1) + 128 * y
        w = 256 * math.exp(w)
        h = 256 * math.exp(h)
        tag = table[cid.item()]
        print(tag)
        draw.rectangle((x - w/2, y - h/2, x + w/2, y + h/2), outline='red')
        draw.text((x-w/2, y-h/2), 'lv4'+tag, font=fnt, fill=(255, 0, 0, 128))


im.show()

