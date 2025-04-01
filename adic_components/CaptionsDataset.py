from torch.utils.data import Dataset
import json
from loguru import logger
import os
from PIL import Image
from torchvision.transforms import v2

default_transform = v2.Compose([
        v2.Resize((224, 224)),
        #v2.RandomCrop((224, 224), pad_if_needed=True, padding_mode='symmetric'),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CaptionDataset(Dataset):
    def __init__(self, images_dir: str, json_path: str, transform=None):
        logger.trace("Initializing CaptionDataset, with images_dir: {}, json_path: {}", images_dir, json_path)
        self.images_dir = images_dir
        self.transform = transform
        annotations = json.load(open(json_path, 'r'))
        self.img_paths = {}
        self.img_cache = {}
        logger.trace("Loading annotations from {}", json_path)
        logger.trace("Loading images")
        for imgdata in annotations['images']:
            id = imgdata['id']
            path = os.path.join(self.images_dir, imgdata['file_name'])
            self.img_paths[id] = path
        logger.info("Loaded {} images", len(self.img_paths))
        logger.trace("Loading captions")
        self.captions = []
        self.image_to_caption = {}
        for imgdata in annotations['annotations']:
            id = imgdata['image_id']
            caption = imgdata['caption']
            assert len(caption) > 0, "Caption is empty"
            assert id in self.img_paths, "Image ID not found in img_paths"
            self.captions.append((self.img_paths[id], caption))
            inserted_index = len(self.captions) - 1
            if id not in self.image_to_caption:
                self.image_to_caption[id] = []
            self.image_to_caption[id].append(inserted_index)
        logger.trace("Loaded {} captions", len(self.captions))

    def __getitem__(self, index):
        img_path, caption = self.captions[index]
        return self.load_image(img_path), caption

    def __len__(self):
        return len(self.captions)

    def load_image(self, img_path):
        logger.trace("Loading image from {}", img_path)
        if img_path in self.img_cache:
            logger.trace("Image found in cache")
            return self.img_cache[img_path]
        else:
            logger.trace("Image not found in cache, loading from disk")
            img = Image.open(img_path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            self.img_cache[img_path] = img
            return img