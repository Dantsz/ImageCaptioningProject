from torch.utils.data import Dataset
import json
from loguru import logger
import os
from PIL import Image
from torchvision.transforms import v2
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer

default_transform = v2.Compose([
        v2.Resize((224, 224)),
        #v2.RandomCrop((224, 224), pad_if_needed=True, padding_mode='symmetric'),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentation_train_transform = v2.Compose([
        v2.Resize(256),
        v2.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmentation_test_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop((224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

default_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
default_tokenizer.pad_token = default_tokenizer.eos_token
def train_collate_fn(batch, tokenizer=default_tokenizer):
    """
    Custom collate function for training.
    Pads the captions within the batch to the maximum length of the batch.

    Args:
        batch (_type_): batch of data from the dataset
        tokenizer (_type_): tokenizer to use for padding the captions

    Returns:
        _type_: batch of images and padded caption
    """
    # pad_sequence pads to max length in batch
    images, captions = zip(*batch)  # unzip the list of tuples
    captions = pad_sequence(list(captions), batch_first=True, padding_value=tokenizer.pad_token_id)
    return torch.stack(images), captions

def test_collate_fn(batch, tokenizer=default_tokenizer):
    """
    Custom collate function for testing.
    Pads the captions within the batch to the maximum length of the batch, and returns the original images and captions as list.

    Args:
        batch (_type_): batch of data from the dataset
        tokenizer (_type_): tokenizer to use for padding the captions

    Returns:
        _type_: batch of: images, padded captions, original images, and original captions
    """
    # pad_sequence pads to max length in batch
    images, captions, org_images, org_captions = zip(*batch)  # unzip the list of tuples
    captions = pad_sequence(list(captions), batch_first=True, padding_value=tokenizer.pad_token_id)
    return torch.stack(images), captions, org_images, org_captions

def add_bos_eos(token_ids: torch.Tensor, bos_token_id: int, eos_token_id: int) -> torch.Tensor:
    bos = torch.full((token_ids.size(0), 1), bos_token_id, dtype=token_ids.dtype, device=token_ids.device)
    eos = torch.full((token_ids.size(0), 1), eos_token_id, dtype=token_ids.dtype, device=token_ids.device)
    return torch.cat([bos, token_ids, eos], dim=1)

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
class CaptionDatasetEager(Dataset):
    '''
    Same as CaptionDataset, but tokenizes the captions eagerly.
    '''
    def __init__(self, images_dir: str, json_path: str, transform=None, tokenizer=None):
        logger.trace("Initializing CaptionDataset, with images_dir: {}, json_path: {}", images_dir, json_path)
        self.images_dir = images_dir
        self.transform = transform
        annotations = json.load(open(json_path, 'r'))
        self.img_paths = {}
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
            tokenized_captions = tokenizer(caption, padding=True, return_tensors="pt", add_special_tokens=True).input_ids
            tokenized_captions = add_bos_eos(tokenized_captions, tokenizer.bos_token_id, tokenizer.eos_token_id)
            tokenized_captions = tokenized_captions.squeeze(0)
            assert id in self.img_paths, "Image ID not found in img_paths"
            self.captions.append((self.img_paths[id], tokenized_captions))
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
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

class CaptionDatasetValidation(Dataset):
    '''
    Same as the eager, but also keeps the untoknenized captions, in order to compute the METEOR score.
    '''
    def __init__(self, images_dir: str, json_path: str, transform=None, tokenizer=None):
        logger.trace("Initializing CaptionDataset, with images_dir: {}, json_path: {}", images_dir, json_path)
        self.images_dir = images_dir
        self.transform = transform
        annotations = json.load(open(json_path, 'r'))
        self.img_paths = {}
        logger.trace("Loading annotations from {}", json_path)
        logger.trace("Loading images")
        for imgdata in annotations['images']:
            id = imgdata['id']
            path = os.path.join(self.images_dir, imgdata['file_name'])
            self.img_paths[id] = path
        logger.info("Loaded {} images", len(self.img_paths))
        logger.trace("Loading captions")
        self.captions = []
        self.text_captions = []
        self.image_to_caption = {}
        self.caption_index_to_image_id = {}
        for imgdata in annotations['annotations']:
            caption_index = len(self.captions)
            id = imgdata['image_id']
            caption = imgdata['caption']
            assert len(caption) > 0, "Caption is empty"
            tokenized_captions = tokenizer(caption, padding=True, return_tensors="pt", add_special_tokens=True).input_ids
            tokenized_captions = add_bos_eos(tokenized_captions, tokenizer.bos_token_id, tokenizer.eos_token_id)
            tokenized_captions = tokenized_captions.squeeze(0)
            assert id in self.img_paths, "Image ID not found in img_paths"
            self.captions.append((self.img_paths[id], tokenized_captions))
            self.text_captions.append((self.img_paths[id], caption))
            inserted_index = len(self.captions) - 1
            if id not in self.image_to_caption:
                self.image_to_caption[id] = []
            self.caption_index_to_image_id[caption_index] = id
            self.image_to_caption[id].append(inserted_index)
        assert len(self.captions) == len(self.text_captions), "Number of captions and text captions do not match"
        logger.trace("Loaded {} captions", len(self.captions))

    def get_image_id_by_index(self, index):
        """
        Returns the image id for a given caption index.
        """
        return self.caption_index_to_image_id[index]

    def __getitem__(self, index):
        img_path, caption = self.captions[index]
        img_transformed, img = self._load_image(img_path)
        return img_transformed, caption, img, self.text_captions[index][1]

    def __len__(self):
        return len(self.captions)

    def _load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_transformed = None
        if self.transform is not None:
            img_transformed = self.transform(img)
        else:
            img_transformed = img
        return img_transformed, img

class CaptionDatasetPyCOCO(CaptionDatasetValidation):
    '''
    Returns an image and the list of references for that image.
    '''
    def __init__(self, images_dir: str, json_path: str, transform=None, tokenizer=None):
        super().__init__(images_dir, json_path, transform, tokenizer)
        self.image_to_caption_list_values = list(self.image_to_caption.values())
        self.image_to_caption_list_keys = list(self.image_to_caption.keys())

    def __getitem__(self, index):
        captiond_idx = self.image_to_caption_list_values[index]
        captions_text = [self.text_captions[i][1] for i in captiond_idx]
        id = self.image_to_caption_list_keys[index]
        path = self.img_paths[id]
        return self._load_image(path)[0], captions_text

    def __len__(self):
        return len(self.image_to_caption_list_values)