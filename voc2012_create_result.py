from PIL import Image, ImageFont, ImageDraw
import torch
import torchvision
import model
import math
import data
import time
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

class Object:
    def __init__(self, tag, bbox, conf):
        self.tag = tag
        self.bbox = bbox
        self.conf = conf

class Detector:
    def __init__(self, model):
        self.model = model
        self.model = self.model.eval()
        self._totensor =torchvision.transforms.ToTensor()

    def __call__(self, x):
        results = []
        
        x = x.resize((256, 256))
        x = self._totensor(x).unsqueeze(0)
        x = self.model(x)

        for xxx in x:
            results += self._extract(xxx[0])

        results = [obj for obj in results if obj.conf > 0.6]

        results = self.non_max_suppress(results)

        return results

    def _extract(self, tensor):
        results = []
        
        grid = tensor.shape[1]  # 15 7 3 1
        grid_size = 512 // (grid + 1)  # 32 64 128 256
        
        for gx in range(grid):
            for gy in range(grid):
                for obj_idx in range(10):
                    conf = tensor[obj_idx, gx, gy].item()
                    if conf < 0:
                        continue
                    tag = table[obj_idx]
                    x = tensor[10, gx, gy]
                    y = tensor[11, gx, gy]
                    w = tensor[12, gx, gy]
                    h = tensor[13, gx, gy]
                    x = (grid_size/2) * (gx + 1 + x)
                    y = (grid_size/2) * (gy + 1 + y)
                    w = grid_size * math.exp(w)
                    h = grid_size * math.exp(h)

                    x1, x2 = int(x - w/2), int(x + w/2)
                    y1, y2 = int(y - h/2), int(y + h/2)     

                    obj = Object(tag, (x1, y1, x2, y2), self.sigmoid(conf))

                    results.append(obj)

        return results

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def non_max_suppress(objects):
        while True:
            new_objects = []
            for obj in objects:
                existed = False
                
                for obj2 in new_objects:
                    if obj.tag != obj2.tag:
                        continue
                    
                    if Detector._is_same_obj(obj, obj2):
                        if obj.conf < obj2.conf:
                            existed = True
                        else:  #override
                            obj2.bbox = obj.bbox
                            obj2.conf = obj.conf
                            existed = True
                        break
                if not existed:
                    new_objects.append(obj)

            if len(new_objects) == len(objects):
                break
            else:
                objects = new_objects
                continue

        return objects
                        
                

    @staticmethod
    def _is_same_obj(obj1, obj2):
        return Detector._iou(obj1.bbox, obj2.bbox) > 0.5

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
    

if __name__ == '__main__':
    fnt = ImageFont.truetype('Pillow/Tests/fonts/arial.ttf', 15)
    
    net = model.MyNet().eval()
    net.load_state_dict(torch.load('model/model'))
    detector = Detector(net)

    im = Image.open('dogcat.jpg').resize((256, 256)).convert('RGB')

    start = time.time()
    results = detector(im)
    stop = time.time()

    
    draw = ImageDraw.Draw(im)
    for obj in results:
        draw.rectangle(obj.bbox, outline='red')
        draw.text(obj.bbox[0:2], obj.tag + ' ' + str(obj.conf), font=fnt, fill=(255, 0, 0, 128))

    print(stop-start)
    im.save('aaaa.png')
    

    

