from typing import Optional
from torch import nn
from adic_components.DyT import DyT
from adic_components.prototype2 import P2EncoderGluer
from torchvision.ops import SqueezeExcitation
import torch
from loguru import logger
from adic_components.prototype2 import P2GPTBlock, P2DecoderCrossAttention
from transformers import GPT2Config
import torch.nn.functional as F
class P3EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_size:int, stride:int = 1, squeeze_channels: int = 16, expansion_factor: int = 4):
        '''
        Args:
            in_channels: The number of input channels
            hidden_size: The inner size of the convolution
            stride: The stride of the convolution
            squeeze_channels: The number of channels to squeeze to
            expansion_factor: How much to multiply the hidden size (the one after the downsampling) by to output
        '''
        super(P3EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.stride = stride
        self.conv1 = nn.Conv2d(in_channels, hidden_size, kernel_size=1, bias=False)# no need for bias because we use batch norm
        self.bn1 = nn.BatchNorm2d(hidden_size)

        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_size)

        self.conv3 = nn.Conv2d(hidden_size, hidden_size * expansion_factor, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_size * expansion_factor)
        self.se = SqueezeExcitation(hidden_size * expansion_factor, squeeze_channels)
        self.act = nn.ReLU(inplace=True)

        if stride != 1:
            self.downsample_identity = nn.Sequential(
                nn.Conv2d(in_channels, hidden_size * expansion_factor, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(hidden_size * expansion_factor)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        indentity = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.se(x)
        if self.stride == 1:
            x += indentity
        else:
            indentity = self.downsample_identity(indentity)
            x += indentity
        x = self.act(x)
        return x

class P3Encoder(nn.Module):
    '''
    Third encoder, second CNN encoder, which is used to encode the input image into a sequence of embeddings.
    https://github.com/JayPatwardhan/ResNet-PyTorch/blob/master/ResNet/ResNet.py helped
    '''

    def __init__(self, input_channels: int, input_width: int, input_height: int, d_model: int, expansion_factor: int = 4, squeeze_channels: int = 16):
        '''
        Args:
            intput_channels: The number of input channels (e.g., 3 for RGB images)
            input_width: The width of the input image
            input_height: The height of the input image
            d_model: The dimension of the model embeddings, which should be the same as the input embeddings of the decoder
            expansion_factor: The expansion factor for subsequent blocks
        '''
        assert input_channels == 3 or input_channels == 1, "Currently only RGB or Grayscale images are supported"
        assert input_width % 16 == 0, "Input width must be a multiple of 16"
        assert input_height % 16 == 0, "Input height must be a multiple of 16"
        super(P3Encoder, self).__init__()
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.squeeze_channels = squeeze_channels
        # derive sequence length from input dimensions
        self.seq_length = (input_width // 16) * (input_height // 16)
        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height
        # CNN blocks
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(3, 64, 32)
        self.layer2 = self._make_layer(8, 128, 64)
        self.layer3 = self._make_layer(24, 256, 128)
        self.layer4 = self._make_layer(32, 512, 192)# final output: 4 * 192 = 768

        self.gluer = P2EncoderGluer(768, self.seq_length, d_model, dropout=0.3)
        self.act = nn.ReLU()

    def _make_layer(self, blocks: int, in_channels: int, hidden_size: int):
        '''
        Args:
            blocks: The number of blocks to make
            in_channels: The number of input channels
            hidden_size: The inner size of the convolution
        '''
        layers = []
        layers.append(P3EncoderBlock(in_channels, hidden_size, expansion_factor=self.expansion_factor, squeeze_channels=self.squeeze_channels, stride=2))
        for _ in range(blocks - 1):
            # hidden_size * self.expansion_factor as input means the number of channels does not modify when goign through these blocks
            layers.append(P3EncoderBlock(hidden_size * self.expansion_factor, hidden_size, expansion_factor=self.expansion_factor, squeeze_channels=self.squeeze_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, input_channels, input_width, input_height)
        Returns:
            A tensor of shape (batch_size, d_model) containing the embeddings
        '''
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (batch_size, d_model, seq_length) -> (batch_size, seq_length, d_modela, this is how the input to cross attention should look like
        x = self.gluer(x)
        return x

class P3DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super(P3DecoderBlock, self).__init__()
        self.cross_attention = P2DecoderCrossAttention(d_model, n_head, dropout=dropout)
        self.norm1 = DyT(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = DyT(d_model)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.cross_attention(x, encoder_output) + x
        x = self.norm1(x)
        residual = x
        x = self.mlp(x) + residual
        x = self.norm2(x)
        return x

class P3Decoder(nn.Module):
    def __init__(self, gpt2_config: GPT2Config, dropout: float = 0.15, cross_attention_blocks: int = 6):
        super(P3Decoder, self).__init__()
        self.gpt2 = P2GPTBlock(gpt2_config)
        self.hidden_size = gpt2_config.n_embd
        self.vocab_size = gpt2_config.vocab_size
        self.norm2 = DyT(self.hidden_size)
        self.catt_blocks = nn.ModuleList([P3DecoderBlock(self.hidden_size, gpt2_config.n_head, self.hidden_size * 4, dropout=dropout) for _ in range(cross_attention_blocks)])
        # Adapter MLP for Q projection before cross-attention
        self.query_adapter = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        # these are the embeddings that that the decoder outputs, the original GPT-2 model uses the same embeddings for input and output
        # but then we can't fine-tune the model without touching the self-attention weights
        # so we use a separate embedding layer for the output
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.gpt2.wte.weight.clone())

    def forward(self, x, encoder_output, attention_mask: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = False) -> torch.Tensor:
        # batch_size, seq_length, d_model = gpt2_output.shape
        x_shape = x.shape
        encoder_output_shape = encoder_output.shape
        assert len(x_shape) == 2, f"Input shape mismatch: {x_shape}, format should be (batch_size, seq_length)"
        assert len(encoder_output_shape) == 3, f"Encoder output shape mismatch: {encoder_output_shape}, format should be (batch_size, seq_length, d_model)"
        assert x_shape[0] == encoder_output_shape[0], f"Batch size mismatch: {x_shape[0]} != {encoder_output_shape[0]}"
        logger.trace("Decoder input shape: {}", x.shape)
        logger.trace("Encoder output shape: {}", encoder_output.shape)
        x = self.gpt2(x, attention_mask=attention_mask, use_cache=use_cache).last_hidden_state # the output of the GPT-2 block is (batch_size, seq_length, d_model)
        x = self.query_adapter(x) + x # this is the Q projection before cross-attention
        logger.trace("Decoder output shape: {}", x.shape)
        for catt_block in self.catt_blocks:
            x = catt_block(x, encoder_output)
        logger.trace("Cross attention output shape: {}", x.shape)
        x = self.norm2(self.mlp(x) + x) # Add and Norm
        x = self.lm_head(x)
        logger.trace("LM head output shape: {}", x.shape)
        return x
class P3ECDEC(nn.Module):
    def __init__(self, input_channels: int, input_width: int, input_height: int, d_model: int, decoder: nn.Module):
        super(P3ECDEC, self).__init__()
        self.encoder = P3Encoder(input_channels, input_width, input_height, d_model)
        self.decoder = decoder

    def forward(self, tokens: torch.Tensor, images: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None, use_cache: Optional[bool] = None) -> torch.Tensor:
        x = self.encoder(images)
        x = self.decoder(tokens, x, attention_mask=attention_mask, use_cache=use_cache)
        return x

    def generate(self, images: torch.Tensor, max_length: int = 16) -> torch.Tensor:
        batch_size = images.shape[0]
        assert batch_size == 1, "Batch size must be 1 for generation, currently"
        tokens = torch.ones(1, 1).long().to(images.device) * self.decoder.gpt2.config.bos_token_id
        while tokens.shape[1] < max_length:
            # get the last token and pass it to the decoder
            x = self.forward(tokens, images)
            # get the last token
            x = torch.argmax(x[:, -1, :], dim=1).unsqueeze(0)
            tokens = torch.cat([tokens, x], dim=1).contiguous()
            if tokens[0, -1] == self.decoder.gpt2.config.eos_token_id:
                break
        return tokens

    def generate_with_beam_search(self, images: torch.Tensor, max_length: int = 25, beam_size: int = 5, temperature: float = 0.5, use_cache: Optional[bool] = True) -> torch.Tensor:
        batch_size = images.shape[0]
        assert batch_size == 1, "Batch size must be 1 for generation, currently"

        # Initial tokens (start with BOS token)
        tokens = torch.ones(1, 1).long().to(images.device) * self.decoder.gpt2.config.bos_token_id

        # Beam initialization: list of (tokens, score)
        beams = [([self.decoder.gpt2.config.bos_token_id], 0.0)] * beam_size
        finished = []

        for _ in range(max_length):
            all_candidates = []

            for tokens, score in beams:
                # Skip beams that are already finished (EOS token generated, which is same as BOS)
                if len(tokens) > 1 and tokens[-1] == self.decoder.gpt2.config.bos_token_id:
                    finished.append((tokens, score))
                    continue

                # Convert tokens to tensor and pass through the model
                input_ids = torch.tensor(tokens, dtype=torch.long, device=images.device).unsqueeze(0)
                logits = self.forward(input_ids, images, use_cache=use_cache)  # [batch_size, seq_len, vocab_size]

                # Apply temperature to logits: divide by temperature
                logits = logits[:, -1, :] / temperature  # Scale logits by temperature

                # Apply log softmax to get log probabilities
                log_probs = F.log_softmax(logits, dim=-1)

                # Top-k candidates (beam_size candidates)
                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)

                # Generate new candidates by appending topk token ids
                for log_prob, token_id in zip(topk_log_probs[0], topk_ids[0]):
                    new_tokens = tokens + [token_id.item()]
                    new_score = score + log_prob.item()  # Accumulate score (log-probability)
                    all_candidates.append((new_tokens, new_score))

            # Select the top beams based on their scores
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            # Stop early if all beams have finished (no tokens added after BOS)
            if all(len(t[0]) > 1 and t[0][-1] == self.decoder.gpt2.config.bos_token_id for t in beams):
                break

        # Add any unfinished beams to the finished list
        finished += [b for b in beams if len(b[0]) > 1 and b[0][-1] == self.decoder.gpt2.config.bos_token_id]

        # Sort beams by their score and return the best one
        finished = sorted(finished, key=lambda x: x[1], reverse=True)

        # Return the best sequence (highest score)
        best_tokens = finished[0][0] if finished else beams[0][0]
        return torch.tensor(best_tokens).unsqueeze(0).to(images.device)


if __name__ == "__main__":
    print(f'P3 Encoder has: {sum(p.numel() for p in P3Encoder(3, 224, 224, 768).parameters())} parameters')
    # Example usage
    encoder = P3Encoder(input_channels=3, input_width=224, input_height=224, d_model=768)
    # Create a random input tensor
    x = torch.randn(5, 3, 224, 224)  # (batch_size, channels, height, width)

    # Forward pass through the encoder
    encoded_x = encoder(x)
    print("Encoded shape:", encoded_x.shape)