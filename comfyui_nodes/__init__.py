
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from adic_components.prototype2 import P2GPTBlock
from adic_components.prototype3 import P3ECDEC, P3Decoder, LoRAdLMHead
from adic_components.CaptionsDataset import augmentation_test_transform, default_tokenizer

class P3ECDECCheckpointLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_path": ("STRING", {"default": "path/to/p3ecdec.pth"}),
                "input_channels": ("INT", {"default": 3}),
                "input_width": ("INT", {"default": 224}),
                "input_height": ("INT", {"default": 224}),
                "d_model": ("INT", {"default": 768}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "Custom"

    def __init__(self):
        self.loaded_model = None  # Caches the model so we don't reload each time

    def load(self, ckpt_path, input_channels, input_width, input_height, d_model):
        print(f"[P3ECDEC Loader] Loading from {ckpt_path}")
        
        gpt2_model_pretrained = GPT2Model.from_pretrained('gpt2')
        # Get model config to know vocab size and hidden size
        config = GPT2Config.from_pretrained('gpt2')
        hidden_size = config.n_embd
        gpt2_model = P2GPTBlock(config)
        gpt2_model.load_state_dict(gpt2_model_pretrained.state_dict(), strict=False)
        decoder = P3Decoder(config)
        decoder.gpt2 = gpt2_model
        #freeze gpt2, for correct calculation of parameters
        for param in gpt2_model.parameters():
            param.requires_grad = False
        encodeco = P3ECDEC(3, 224, 224, hidden_size, decoder)
        model = encodeco
        
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        self.loaded_model = model
        return (model,)

class P3ECDECNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),  # The model instance passed from upstream node
                "image": ("IMAGE",), 
                "max_length": ("INT", {"default": 16, "min": 1, "max": 64}),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "Custom"

    def generate(self, model, image, max_length):
        if model is None:
            raise RuntimeError("No model passed to P3ECDECNode")

        if image.shape[0] != 1:
            raise ValueError("Batch size must be 1 for generation")

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)

        print(f"[P3ECDECNode] Generating with model {model.__class__.__name__} and image shape {image.shape}")
        image = image.permute(0, 3, 1, 2).contiguous()
        image = augmentation_test_transform(image)

        device = next(model.parameters()).device
        image = image.to(device)

        with torch.no_grad():
            tokens = model.generate(image, max_length=max_length)

        generated = default_tokenizer.batch_decode(tokens.cpu().tolist())
        return (generated,)

NODE_CLASS_MAPPINGS = {
    "P3ECDECCheckpointLoader": P3ECDECCheckpointLoader,
    "P3ECDECNode": P3ECDECNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "P3ECDECCheckpointLoader": "Load P3ECDEC Checkpoint",
    "P3ECDECNode": "P3ECDEC Generate",
}
