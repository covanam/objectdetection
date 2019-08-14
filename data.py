import torch
import random
import PIL
import torchvision
import xml.etree.ElementTree as Et


class Dataset(torch.utils.data.Dataset):
    class_tags = {'bicycle': 1,
                  'bus': 2,
                  'car': 3,
                  'cat': 4,
                  'cow': 5,
                  'dog': 6,
                  'horse': 7,
                  'motorbike': 8,
                  'person': 9,
                  'sheep': 10}

    class Object:
        def __init__(self, tag, bbox):
            self.tag = tag
            self.bbox = bbox  # xmin, ymin, xmax, ymax

    def __init__(self, file_dir, size=(224, 224), grid_size=32, data_arg=True):
        with open(file_dir) as file:
            self.im_list = [line.strip() for line in file]
        self.data_arg = data_arg
        self.size = size
        self.grid_size = grid_size
        self.__preprocess()

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im_name = self.im_list[idx]

        # read image
        im = PIL.Image.open(self.__im_path(im_name))
        im.load()

        # read objects
        objects = self.__xml_parse(self.__xml_path(im_name))

        # perform data argumentation
        if self.data_arg:
            im = self.__colorjitter(im)

            # 50% chance to flip image
            god_will = random.randrange(2)
            if god_will:
                im, objects = self.__flip(im, objects)

            # 50% chance to crop image
            god_will = random.randrange(2)
            if god_will:
                im, objects = self.__crop(im, objects)

        # resize image to the expected size:
        im, objects = self.__resize(im, objects, self.size)

        # construct output tensor used to train neural nework
        im, out = self.__construct_out_tensor(im, objects, self.grid_size)

        im = self.__totensor(im)

        return im, out

    @staticmethod
    def __have_interested_object(objects):
        for obj in objects:
            if obj.tag in Dataset.class_tags:
                return True
        return False

    def __preprocess(self):
        """eleminate all image that doesn't contain a interested object"""
        processed_list = []

        for im_name in self.im_list:
            objects = Dataset.__xml_parse(Dataset.__xml_path(im_name))
            if self.__have_interested_object(objects):
                processed_list.append(im_name)

        self.im_list = processed_list

    # argumentation:
    __colorjitter = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    @staticmethod
    def __flip(im, objects):
        """flip image, nothing much"""
        im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        width, height = im.size

        for object in objects:
            xmin, ymin, xmax, ymax = object.bbox
            xmin, xmax= width - xmax, width - xmin
            object.bbox = (xmin, ymin, xmax, ymax)

        return im, objects

    @staticmethod
    def __is_inside(bbox, crop):
        """check if bbox is still inside a crop, used in __crop"""
        left, upper, right, lower = crop
        xmin, ymin, xmax, ymax = bbox
        if left < (xmin + xmax) // 2 < right and upper < (ymin + ymax) // 2 < lower:
            return True
        return False

    @staticmethod
    def __crop(im, objects):
        """randomly crop image"""
        width, height = im.size

        left = random.randrange(width // 10)
        upper = random.randrange(height // 10)
        right = width - 1 - random.randrange(width // 10)
        lower = height - 1 - random.randrange(height // 10)

        im = im.crop((left, upper, right, lower))

        objects = [obj for obj in objects if Dataset.__is_inside(obj.bbox, (left, upper, right, lower))]
        for object in objects:
            xmin, ymin, xmax, ymax = object.bbox
            object.bbox = xmin - left, ymin - upper, xmax - left, ymax - upper

        return im, objects

    # file processing

    @staticmethod
    def __im_path(im_name):
        return 'VOC2012/JPEGImages/' + im_name + '.jpg'

    @staticmethod
    def __xml_path(im_name):
        return 'VOC2012/Annotations/' + im_name + '.xml'

    @staticmethod
    def __xml_parse(xml_dir):
        root = Et.parse(xml_dir).getroot()
        objects = []
        for obj in root.findall('object'):
            tag = obj.find('name').text
            bndbox = obj.find('bndbox')
            bbox = (int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text))
            objects.append(Dataset.Object(tag, bbox))

        return objects

    # image processing before feeding to network
    __totensor = torchvision.transforms.ToTensor()

    @staticmethod
    def __resize(im, objects, size):
        scale_x = size[0] / im.size[0]
        scale_y = size[1] / im.size[1]

        im = im.resize(size, PIL.Image.ANTIALIAS)

        for object in objects:
            xmin, ymin, xmax, ymax = object.bbox
            xmin = int(xmin * scale_x)
            ymin = int(ymin * scale_y)
            xmax = int(xmax * scale_x)
            ymax = int(ymax * scale_y)
            object.bbox = xmin, ymin, xmax, ymax

        return im, objects

    @staticmethod
    def __construct_out_tensor(im, objects, grid_size=32):
        width, height = im.size
        out_w = width // grid_size
        out_h = height // grid_size

        out = torch.zeros(out_w, out_h, dtype=torch.long)

        # assigning ignored grids
        for object in objects:
            tag = object.tag
            tag_index = Dataset.class_tags.get(tag)
            if tag_index:
                xmin, ymin, xmax, ymax = object.bbox

                x1 = xmin // grid_size
                x2 = xmax // grid_size + 1
                y1 = ymin // grid_size
                y2 = ymax // grid_size + 1

                if x1 >= out_w:
                    x1 = out_w - 1
                if x1 < 0:
                    x1 = 0
                if x2 == 0:
                    x2 = 1
                if y1 >= out_h:
                    y1 = out_h - 1
                if y1 < 0:
                    y1 = 0
                if y2 == 0:
                    y2 = 1

                out[x1:x2, y1:y2] = -1

        # assigning object grids
        for object in objects:
            tag = object.tag
            tag_index = Dataset.class_tags.get(tag)
            if tag_index:
                xmin, ymin, xmax, ymax = object.bbox
                x = (xmin + xmax) // 2 // grid_size
                y = (ymin + ymax) // 2 // grid_size
                out[x, y] = tag_index

        return im, out


table = {
    -1: 'ignored',
    0: 'ground',
    1: 'bicycle',
    2: 'bus',
    3: 'car',
    4: 'cat',
    5: 'cow',
    6: 'dog',
    7: 'horse',
    8: 'motorbike',
    9: 'person',
    10: 'sheep'}
if __name__ == '__main__':
    dataset = Dataset('VOC2012/ImageSets/Main/test.txt', data_arg=False, size=(224, 224))
    from PIL import Image, ImageFont, ImageDraw
    fuckkk = torchvision.transforms.ToPILImage()

    im, out = dataset[0]

    im = fuckkk(im)

    draw = ImageDraw.Draw(im)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
    for gx in range(out.shape[0]):
        for gy in range(out.shape[1]):
            draw.text((gx*32, gy*32), str(int(out[gx, gy].item())), font=fnt, fill=(255, 0, 0, 128))

    im.show()