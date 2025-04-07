from torch.utils.data import Dataset
import json
from loguru import logger
import os
from PIL import Image
from torchvision.transforms import v2
import torch

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
class CaptionDatasetTrain(Dataset):
    """
        Dataset for training, with cached images and tokenized captions.
    """
    def __init__(self, images_dir: str, json_path: str, transform=None, tokenizer=None, output_device=None):
        logger.trace("Initializing CaptionDataset, with images_dir: {}, json_path: {}", images_dir, json_path)
        self.images_dir = images_dir
        self.transform = transform
        self.tokenizer = tokenizer  # Now you have access to the tokenizer
        self.output_device = output_device
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
        self.max_length = 0
        self.image_to_caption = {}
        for imgdata in annotations['annotations']:
            caption = imgdata['caption']
            tokenized_caption = self.tokenizer(caption, truncation=False, padding=True,
                                               return_tensors="pt", add_special_tokens=True).input_ids
            length = tokenized_caption.shape[1]
            if length > self.max_length:
                self.max_length = length

        for imgdata in annotations['annotations']:
            id = imgdata['image_id']
            caption = imgdata['caption']
            assert len(caption) > 0, "Caption is empty"
            assert id in self.img_paths, "Image ID not found in img_paths"

            # Tokenizing captions here with padding but not truncating
            tokenized_caption = self.tokenizer(caption, truncation=False, padding=True,
                                               return_tensors="pt", add_special_tokens=True).input_ids

            # Add BOS and EOS tokens
            tokenized_caption = self.add_bos_eos(tokenized_caption, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)

            # Make sure all captions are padded up to max_length
            tokenized_caption = tokenized_caption.squeeze(0)  # Remove unnecessary batch dimension
            assert tokenized_caption.shape[0] == self.max_length + 2, f"Tokenized caption shape mismatch, expected {(self.max_length + 2,)}, got {tokenized_caption.shape}"

            self.captions.append((self.img_paths[id], tokenized_caption.to(self.output_device)))
            inserted_index = len(self.captions) - 1
            if id not in self.image_to_caption:
                self.image_to_caption[id] = []
            self.image_to_caption[id].append(inserted_index)
        assert self.max_length > 0, "Max length is zero"
        self.max_length = self.max_length + 2  # Adding 2 for BOS and EOS tokens
        logger.trace("Loaded {} captions", len(self.captions))

    def add_bos_eos(self, token_ids: torch.Tensor, bos_token_id: int, eos_token_id: int) -> torch.Tensor:
        """
        Add BOS and EOS tokens to the tokenized caption.
        """
        bos = torch.full((token_ids.size(0), 1), bos_token_id, dtype=token_ids.dtype, device=token_ids.device)
        eos = torch.full((token_ids.size(0), 1), eos_token_id, dtype=token_ids.dtype, device=token_ids.device)
        return torch.cat([bos, token_ids, eos], dim=1)
    def __getitem__(self, index):
        """
        Return the image and the tokenized caption.
        """
        img_path, tokenized_caption = self.captions[index]
        img = self.load_image(img_path)
        return img, tokenized_caption # Removing the batch dimension for the tokenized caption

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
                img = self.transform(img).to(self.output_device)
            self.img_cache[img_path] = img
            return img