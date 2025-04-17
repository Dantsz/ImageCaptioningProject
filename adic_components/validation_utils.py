import evaluate
meteor = evaluate.load("meteor")
import torch
from typing import List
from torch.nn import Module
from PIL import Image
from adic_components.CaptionsDataset import default_transform

def compute_meteor_score(predictions, references) -> float:
    return meteor.compute(predictions=predictions, references=references)["meteor"]

def evaluate_model(model: Module, image: Image, original_caption: str, tokenizer, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> float:
    """
    Evaluates the model's performance on a single image and its original caption.

    Args:
        model (Module): The model to evaluate.
        image (Image): The input image.
        original_caption (str): The original caption for the image.
        tokenizer: The tokenizer used for decoding the generated caption.
        device (torch.device): The device to run the model on (default is GPU if available).

    Returns:
        float: The METEOR score of the model's prediction against the original caption.
    """
    img_pixel_values = default_transform(image).unsqueeze(0)
    img_pixel_values = img_pixel_values.to(device)
    generated_caption = model.generate(img_pixel_values)  # Placeholder for actual generation logic
    generated_caption = tokenizer.batch_decode(generated_caption.cpu(), skip_special_tokens=True)[0]

    # Compute METEOR score
    meteor_score = compute_meteor_score([generated_caption], [original_caption])

    return meteor_score