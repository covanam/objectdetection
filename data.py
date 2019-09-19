import torch
import random
import PIL
import torchvision
import xml.etree.ElementTree as Et
import math

__all__ = ['Dataset', 'TensorDataset']


class Dataset(torch.utils.data.Dataset):
    interested_classes = {
        'bicycle': 0,
        'bus': 1,
        'car': 2,
        'cat': 3,
        'cow': 4,
        'dog': 5,
        'horse': 6,
        'motorbike': 7,
        'person': 8,
        'sheep': 9
    }

    class Object:
        def __init__(self, tag, bbox):
            self.tag = tag
            self.bbox = bbox  # xmin, ymin, xmax, ymax

        def area(self):
            w = self.bbox[2] - self.bbox[0]
            h = self.bbox[3] - self.bbox[1]
            return w * h

        def x(self):
            return (self.bbox[2] + self.bbox[0]) // 2

        def y(self):
            return (self.bbox[3] + self.bbox[1]) // 2

        def w(self):
            return self.bbox[2] - self.bbox[0]

        def h(self):
            return self.bbox[3] - self.bbox[1]

    def __init__(self, file_dir, data_arg=True):
        with open(file_dir) as file:
            self.im_list = [line.strip() for line in file]
        self.data_arg = data_arg
        self.__preprocess()

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im_name = self.im_list[idx]

        # read image
        im = PIL.Image.open(self.__im_path(im_name))
        im = im.convert('RGB')

        # read objects
        objects = self.__xml_parse(self.__xml_path(im_name))

        if self.data_arg:
            # sample a square part of the image
            im, objects = self.__sample(im, objects)

            # color jittering
            im = self.__colorjitter(im)

            # randomly flipping image
            god_will = random.randrange(2)
            if god_will:
                im, objects = self.__flip(im, objects)

        # resize image to the expected size:
        im, objects = self.__resize(im, objects, (512, 512))

        return im, objects

    @staticmethod
    def __have_interested_object(objects):
        for obj in objects:
            if obj.tag in Dataset.interested_classes:
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
    __colorjitter = torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    @staticmethod
    def __flip(im, objects):
        """flip image, nothing much"""
        im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        width, height = im.size

        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox
            xmin, xmax = width - xmax, width - xmin
            obj.bbox = (xmin, ymin, xmax, ymax)

        return im, objects

    @staticmethod
    def __is_inside(bbox, crop):
        """helper function for __crop below"""
        left, upper, right, lower = crop
        xmin, ymin, xmax, ymax = bbox
        if left < (xmin + xmax) // 2 < right and upper < (ymin + ymax) // 2 < lower:
            return True
        return False

    @staticmethod
    def __crop(im, objects, box):
        width, height = im.size

        left, upper, right, lower = box

        im = im.crop((left, upper, right, lower))

        objects = [obj for obj in objects if Dataset.__is_inside(obj.bbox, (left, upper, right, lower))]
        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox

            xmin = Dataset._saturate_cast(xmin - left, width)
            ymin = Dataset._saturate_cast(ymin - upper, height)
            xmax = Dataset._saturate_cast(xmax - left, width)
            ymax = Dataset._saturate_cast(ymax - upper, height)

            obj.bbox = xmin, ymin, xmax, ymax

        return im, objects

    @staticmethod
    def _saturate_cast(x, m):
        if x < 0:
            return 0
        if x > m - 1:
            return m - 1
        return x

    @staticmethod
    def __pad(im, objects):
        """ randomly pad image to square with random color """
        # calculate what size padded image should be
        new_size = max(im.size)

        # randomly pick a color for padding
        pad_color = random.randrange(256), random.randrange(256), random.randrange(256)

        # initiliza the new image
        padded_im = PIL.Image.new('RGB', (new_size, new_size), pad_color)

        # randomly choose paste location:
        pos_x = random.randrange(0, new_size - im.size[0] + 1)
        pos_y = random.randrange(0, new_size - im.size[1] + 1)

        # copy original image to new square image
        padded_im.paste(im, (pos_x, pos_y))

        # calculate new bounding box for objects
        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox
            obj.bbox = xmin + pos_x, ymin + pos_y, xmax + pos_x, ymax + pos_y

        return padded_im, objects

    @staticmethod
    def __sample(im, objects):
        """take a random rectangle in the image, may be outside of image too"""
        width, height = im.size
        short = min(width, height)
        long = max(width, height)

        # randomly pick sample position and size
        sample_size = random.randrange(short, long + 1)
        sample_pos = random.randrange(0, long - sample_size + 1)

        # calculate the bounding box of sampled rectangle
        if height < width:
            upper, left = 0, sample_pos
            lower, right = height, sample_pos + sample_size
        else:
            upper, left = sample_pos, 0
            lower, right = sample_pos + sample_size, width

        im, objects = Dataset.__crop(im, objects, (left, upper, right, lower))
        im, objects = Dataset.__pad(im, objects)

        return im, objects

    @staticmethod
    def __im_path(im_name):
        """ construct full image directory """
        return 'VOC2012/JPEGImages/' + im_name + '.jpg'

    @staticmethod
    def __xml_path(im_name):
        """ construct full xml directory """
        return 'VOC2012/Annotations/' + im_name + '.xml'

    @staticmethod
    def __xml_parse(xml_dir):
        """ read xml file and return a list of objects """

        root = Et.parse(xml_dir).getroot()

        # container
        objects = []

        for obj in root.findall('object'):
            tag = obj.find('name').text

            # ignore objects that we are not interested in
            if tag not in Dataset.interested_classes:
                continue

            bndbox = obj.find('bndbox')
            bbox = (int(bndbox.find('xmin').text),
                    int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text),
                    int(bndbox.find('ymax').text))
            objects.append(Dataset.Object(tag, bbox))

        return objects

    @staticmethod
    def __resize(im, objects, size):
        scale_x = size[0] / im.size[0]
        scale_y = size[1] / im.size[1]

        im = im.resize(size, PIL.Image.ANTIALIAS)

        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox
            xmin = int(xmin * scale_x)
            ymin = int(ymin * scale_y)
            xmax = int(xmax * scale_x)
            ymax = int(ymax * scale_y)
            obj.bbox = xmin, ymin, xmax, ymax

        return im, objects


class TensorDataset(torch.utils.data.Dataset):
    """ wrapper of above dataset, but return torch.tensor """

    def __init__(self, dataset):
        self.dataset = dataset
        self._totensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        im, objects = self.dataset[item]

        x = self._totensor(im)
        target = self._construct_target_tensor(objects)

        return x, target

    def _construct_target_tensor(self, objects):
        target = []

        # initilize container for target tensors
        for level in range(4):  # 4 levels: 15x15, 7x7, 3x3, 1x1
            size = 16 // 2 ** level - 1
            target.append(torch.zeros((50, size, size), dtype=torch.float))

        # set which grid will be ignored
        # (grid that does overlap with object, but not the best grid
        # that overlap will be ignored
        for obj in objects:
            self._assign_ignore_zone(obj, target)

        # set which grid has an object
        for obj in objects:
            self._assign_obj(obj, target)

        return target

    def _assign_ignore_zone(self, obj, target):
        level, sub_level = self._assign_level(obj)
        tag_id = self._encode_class(obj.tag)
        gx, gy = self._assign_grid(obj, level)
        grid = 16 // 2 ** level - 1  # 15 7 3 1

        for ix in range(max(0, gx - 1), min(grid, gx + 2)):
            for iy in range(max(0, gy - 1), min(grid, gy + 2)):
                if self._iou(obj.bbox, self._grid_bbox(ix, iy, level)) > 0:
                    target[level][tag_id, ix, iy] = -1

        if sub_level != -1:
            gx, gy = self._assign_grid(obj, sub_level)
            target[sub_level][tag_id, gx, gy] = -1

    @staticmethod
    def _grid_bbox(gx, gy, level):
        """ calculate the bounding box of a grid cell """
        grid_size = 64 * 2 ** level
        xmin = (grid_size // 2) * gx
        ymin = (grid_size // 2) * gy
        xmax = xmin + grid_size
        ymax = ymin + grid_size

        return xmin, ymin, xmax, ymax

    @staticmethod
    def _iou(bbox1, bbox2):
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bb2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        iou = intersection_area / (bb1_area + bb2_area - intersection_area)

        return iou

    def _assign_obj(self, obj, target):
        """ assign confident score, bounding box of object into target tensor """
        # calculate which grid does this object belong to
        level = self._assign_level(obj)[0]
        gx, gy = self._assign_grid(obj, level)

        # read object's properties
        tag_id = self._encode_class(obj.tag)

        x = obj.x()
        y = obj.y()
        w = obj.w()
        h = obj.h()

        # calculate actual value put into tensor
        gridsize = 64 * 2 ** level  # 32 64 128 256
        rx, ry = self._calc_relative_loc(gx, gy, x, y, level)
        logw, logh = math.log(w / gridsize), math.log(h / gridsize)

        target[level][10 + 4 * tag_id, gx, gy] = rx
        target[level][11 + 4 * tag_id, gx, gy] = ry
        target[level][12 + 4 * tag_id, gx, gy] = logw
        target[level][13 + 4 * tag_id, gx, gy] = logh

        target[level][tag_id, gx, gy] = 1

    @staticmethod
    def _assign_level(obj):
        area = obj.area()
        if area < 8192:  # 0 -> 2x32x32
            if area > 6144:
                return 0, 1
            else:
                return 0, -1

        if area < 32768:  # 2x32x32 -> 2x64x64
            if area > 24576:
                return 1, 2
            if area < 12288:
                return 1, 0
            return 1, -1

        if area < 131072:  # 2x64x64 -> 2x128x128
            if area > 98304:
                return 2, 3
            if area < 49152:
                return 2, 1
            return 2, -1

        if area < 196608:
            return 3, 2
        return 3, -1

    @staticmethod
    def _assign_grid(obj, level):
        """ hi """
        if level == 3:
            return 0, 0
        grid = 16 // (2 ** level) - 1  # 15, 7, 3, 1
        micro_grid = 32 // 2 ** level  # 32 16 8 4

        grid_x = obj.x() // (512 // micro_grid)
        grid_x = (grid_x + 1) // 2 - 1
        if grid_x < 0:
            grid_x = 0
        if grid_x > grid - 1:
            grid_x = grid - 1

        grid_y = obj.y() // (512 // micro_grid)
        grid_y = (grid_y + 1) // 2 - 1
        if grid_y < 0:
            grid_y = 0
        if grid_y > grid - 1:
            grid_y = grid - 1

        return grid_x, grid_y

    @staticmethod
    def _calc_relative_loc(grid_x, grid_y, x, y, level):
        grid_size = 64 * 2 ** level
        relative_x = x / (grid_size / 2) - (grid_x + 1)
        relative_y = y / (grid_size / 2) - (grid_y + 1)

        return relative_x, relative_y

    @staticmethod
    def _encode_class(tag):
        return Dataset.interested_classes[tag]


def _test():
    from PIL import Image, ImageFont, ImageDraw
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
    dataset = Dataset('VOC2012/ImageSets/Main/test.txt', data_arg=True)

    tdataset = TensorDataset(dataset)

    im, out = tdataset[0]
    im = torchvision.transforms.ToPILImage()(im)

    draw = ImageDraw.Draw(im)
    fnt = ImageFont.truetype('arial.ttf', 15)

    for level in range(4):
        gg = 16 // 2 ** level - 1
        grid = 64 * 2 ** level
        for gx in range(gg):
            for gy in range(gg):
                for i in range(10):
                    conf = out[level][i, gx, gy]
                    if conf.item() != 1:
                        continue

                    x = out[level][10 + 4 * i, gx, gy].item()
                    y = out[level][11 + 4 * i, gx, gy].item()
                    w = out[level][12 + 4 * i, gx, gy].item()
                    h = out[level][13 + 4 * i, gx, gy].item()
                    x = (grid // 2) * (1 + gx + x)
                    y = (grid // 2) * (1 + gy + y)
                    w = grid * math.exp(w)
                    h = grid * math.exp(h)

                    tag = table[i] + str(conf.item())

                    draw.rectangle((x - w / 2, y - h / 2, x + w / 2, y + h / 2), outline='red')
                    draw.text((x - w / 2, y - h / 2), tag, font=fnt, fill=(255, 0, 0, 128))

    im.show()


if __name__ == '__main__':
    _test()
