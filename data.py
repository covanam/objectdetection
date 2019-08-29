import torch
import random
import PIL
import torchvision
import xml.etree.ElementTree as Et
import math


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

    class Holder:
        def __init__(self, data):
            self.data = data

    def __init__(self, file_dir, size=(256, 256), data_arg=True):
        with open(file_dir) as file:
            self.im_list = [line.strip() for line in file]
        self.data_arg = data_arg
        self.size = size
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

        # perform data argumentation
        if self.data_arg:
            im = self.__colorjitter(im)

            god_will = random.randrange(2)
            if god_will:
                im = self.__pad(im)

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

        # convert to tensor to feed to network for training
        im = self.__totensor(im)

        target = self._construct_target_tensor(objects)

        return im, target

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
    __colorjitter = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)

    @staticmethod
    def __flip(im, objects):
        """flip image, nothing much"""
        im = im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        width, height = im.size

        for object in objects:
            xmin, ymin, xmax, ymax = object.bbox
            xmin, xmax = width - xmax, width - xmin
            object.bbox = (xmin, ymin, xmax, ymax)

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
    def __crop(im, objects):
        """randomly crop image"""
        width, height = im.size

        left = random.randrange(width // 10)
        upper = random.randrange(height // 10)
        right = width - 1 - random.randrange(width // 10)
        lower = height - 1 - random.randrange(height // 10)

        im = im.crop((left, upper, right, lower))

        objects = [obj for obj in objects if Dataset.__is_inside(obj.bbox, (left, upper, right, lower))]
        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox
            obj.bbox = xmin - left, ymin - upper, xmax - left, ymax - upper

        return im, objects

    @staticmethod
    def __pad(im):
        new_size = max(im.size)
        pad_color = random.randrange(256), random.randrange(256), random.randrange(256)
        padded_im = PIL.Image.new('RGB', (new_size, new_size), pad_color)
        padded_im.paste(im, (0, 0))

        return padded_im

    @staticmethod
    def __im_path(im_name):
        return 'VOC2012+2007/JPEGImages/' + im_name + '.jpg'

    @staticmethod
    def __xml_path(im_name):
        return 'VOC2012+2007/Annotations/' + im_name + '.xml'

    @staticmethod
    def __xml_parse(xml_dir):
        root = Et.parse(xml_dir).getroot()
        objects = []
        for obj in root.findall('object'):
            tag = obj.find('name').text
            if tag not in Dataset.interested_classes:
                continue
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

        for obj in objects:
            xmin, ymin, xmax, ymax = obj.bbox
            xmin = int(xmin * scale_x)
            ymin = int(ymin * scale_y)
            xmax = int(xmax * scale_x)
            ymax = int(ymax * scale_y)
            obj.bbox = xmin, ymin, xmax, ymax

        return im, objects

    def _construct_target_tensor(self, objects):
        target = []

        for level in range(4):  # 4 levels: 15x15, 7x7, 3x3, 1x1
            size = 16 // 2 ** level - 1
            target.append(torch.zeros((15, size, size), dtype=torch.float))

        for obj in objects:
            self._assign_ignore_zone(obj, target)

        for obj in objects:
            self._assign_obj(obj, target)

        self._finalize_target_tensor(target)

        return target

    def _assign_ignore_zone(self, obj, target):
        level, sub_level = self._assign_level(obj)
        tag_id = self._encode_class(obj.tag)
        gx, gy = self._assign_grid(obj, level)
        grid = 16 // 2**level - 1  # 15 7 3 1

        for ix in range(max(0, gx-1), min(grid, gx+2)):
            for iy in range(max(0, gy-1), min(grid, gy+2)):
                if self._iou(obj.bbox, self._grid_bbox(ix, iy, level)) > 0:
                    target[level][tag_id, ix, iy] = -1

        if sub_level != -1:
            gx, gy = self._assign_grid(obj, sub_level)
            target[sub_level][tag_id, gx, gy] = -1

    @staticmethod
    def _grid_bbox(gx, gy, level):
        grid_size = 32 * 2**level
        xmin = (grid_size // 2) * (gx + 1)
        ymin = (grid_size // 2) * (gy + 1)
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
        level = self._assign_level(obj)[0]
        tag_id = self._encode_class(obj.tag)

        grid_x, grid_y = self._assign_grid(obj, level)

        if target[level][14, grid_x, grid_y].item() == 0:  # no object reside here, nice!
            target[level][10, grid_x, grid_y] = obj.bbox[0]
            target[level][11, grid_x, grid_y] = obj.bbox[1]
            target[level][12, grid_x, grid_y] = obj.bbox[2]
            target[level][13, grid_x, grid_y] = obj.bbox[3]
            target[level][14, grid_x, grid_y] = 1  # mark as occupied
        else:  # already have an object
            target[level][10, grid_x, grid_y] = min(target[level][10, grid_x, grid_y].item(), obj.bbox[0])
            target[level][11, grid_x, grid_y] = min(target[level][11, grid_x, grid_y].item(), obj.bbox[1])
            target[level][12, grid_x, grid_y] = max(target[level][12, grid_x, grid_y].item(), obj.bbox[2])
            target[level][13, grid_x, grid_y] = max(target[level][13, grid_x, grid_y].item(), obj.bbox[3])

        target[level][tag_id, grid_x, grid_y] = 1

    def _finalize_target_tensor(self, target):
        for i in range(4):
            grid_size = 32 * 2**i
            for gx in range(16 // 2**i - 1):
                for gy in range(16 // 2**i - 1):
                    if target[i][14, gx, gy] == 0:
                        continue
                    xmin = target[i][10, gx, gy].item()
                    ymin = target[i][11, gx, gy].item()
                    xmax = target[i][12, gx, gy].item()
                    ymax = target[i][13, gx, gy].item()

                    if xmin < 0: xmin = 0
                    if xmax > 256: xmax = 256
                    if ymin < 0: ymin = 0
                    if ymax > 256: ymax = 256

                    x = (xmin + xmax) / 2
                    y = (ymin + ymax) / 2
                    w = xmax - xmin
                    h = ymax - ymin

                    rx, ry = self._cal_relative_loc(gx, gy, x, y, i)

                    target[i][10, gx, gy] = rx
                    target[i][11, gx, gy] = ry
                    target[i][12, gx, gy] = math.log(w / grid_size)
                    target[i][13, gx, gy] = math.log(h / grid_size)

    @staticmethod
    def _assign_level(obj):
        area = obj.area()
        if area < 2048:  # 0 -> 2x32x32
            if area > 1536:
                return 0, 1
            else:
                return 0, -1

        if area < 8192:  # 2x32x32 -> 2x64x64
            if area > 6144:
                return 1, 2
            if area < 3072:
                return 1, 0
            return 1, -1

        if area < 32768:  # 2x64x64 -> 2x128x128
            if area > 24576:
                return 2, 3
            if area < 12288:
                return 2, 1
            return 2, -1

        if area < 49152:
            return 3, 2
        return 3, -1

    @staticmethod
    def _assign_grid(obj, level):
        if level == 3:
            return 0, 0
        grid = 16 // (2 ** level) - 1  # 15, 7, 3, 1
        micro_grid = 32 // 2**level  # 32 16 8 4

        grid_x = obj.x() // (256 // micro_grid)
        grid_x = (grid_x + 1) // 2 - 1
        if grid_x < 0:
            grid_x = 0
        if grid_x > grid - 1:
            grid_x = grid - 1

        grid_y = obj.y() // (256 // micro_grid)
        grid_y = (grid_y + 1) // 2 - 1
        if grid_y < 0:
            grid_y = 0
        if grid_y > grid - 1:
            grid_y = grid - 1

        return grid_x, grid_y

    @staticmethod
    def _cal_relative_loc(grid_x, grid_y, x, y, level):
        grid_size = 32 * 2 ** level
        relative_x = x / (grid_size / 2) - (grid_x + 1)
        relative_y = y / (grid_size / 2) - (grid_y + 1)

        return relative_x, relative_y

    @staticmethod
    def _encode_class(tag):
        return Dataset.interested_classes[tag]
