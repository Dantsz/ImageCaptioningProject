import torch
import copy
from typing import Optional, Tuple, Union
from torch import nn
from transformers import GPT2Model, GPT2Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, _prepare_4d_attention_mask_for_sdpa
from adic_components.DyT import DyT
from loguru import logger
class P2EncoderGluer(nn.Module):
    '''
        Adjusts the output of the encoder to be compatible with the decoder.
    '''
    def __init__(self, encoder_token_dim: int, encoder_seq_length: int, decoder_token_dim: int):
        super(P2EncoderGluer, self).__init__()
        self.proj = nn.Linear(encoder_token_dim, decoder_token_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, encoder_seq_length, decoder_token_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, encoder_token_dim)
        Returns:
            A tensor of shape (batch_size, decoder_token_dim) containing the embeddings
        '''
        x = self.proj(x)
        x = x + self.positional_encoding
        return x

class P2Encoder(nn.Module):
    '''Encoder for the second prototype.
    This time I'm trying to use a normal CNN to generate the embeddings.
    '''

    def __init__(self, input_channels: int, input_width: int, input_height: int, d_model: int):
        '''
        Args:
            intput_channels: The number of input channels (e.g., 3 for RGB images)
            input_width: The width of the input image
            input_height: The height of the input image
            d_model: The dimension of the model embeddings, which should be the same as the input embeddings of the decoder
        '''
        assert input_channels == 3 or input_channels == 1, "Currently only RGB or Grayscale images are supported"
        assert input_width % 16 == 0, "Input width must be a multiple of 16"
        assert input_height % 16 == 0, "Input height must be a multiple of 16"
        super(P2Encoder, self).__init__()
        self.d_model = d_model
        # derive sequence length from input dimensions
        self.seq_length = (input_width // 16) * (input_height // 16)
        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height
        # CNN layers
        self.conv1_1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        # Residual connections
        self.identity1 = nn.Conv2d(input_channels, 64, kernel_size=1, stride=2, padding=0)
        self.identity2 = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)
        self.identity3 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)
        self.identity4 = nn.Conv2d(256, 512, kernel_size=1, stride=2, padding=0)

        # Batch normalization layers for the residuals
        self.bn_res1 = nn.BatchNorm2d(64)
        self.bn_res2 = nn.BatchNorm2d(128)
        self.bn_res3 = nn.BatchNorm2d(256)
        self.bn_res4 = nn.BatchNorm2d(512)

        self.gluer = P2EncoderGluer(512, self.seq_length, d_model)
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: The input tensor of shape (batch_size, input_channels, input_width, input_height)
        Returns:
            A tensor of shape (batch_size, d_model) containing the embeddings
        '''

        identity = self.bn_res1(self.identity1(x))
        x = self.pool(self.act(self.bn1(self.conv1_1(x))))
        x = x + identity

        identity = self.bn_res2(self.identity2(x))
        x = self.pool(self.act(self.bn2(self.conv2_1(x))))
        x = x + identity

        identity = self.bn_res3(self.identity3(x))
        x = self.pool(self.act(self.bn3(self.conv3_1(x))))
        x = x + identity

        identity = self.bn_res4(self.identity4(x))
        x = self.pool(self.act(self.bn4(self.conv4_1(x))))
        x = x + identity

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (batch_size, d_model, seq_length) -> (batch_size, seq_length, d_modela, this is how the input to cross attention should look like
        x = self.gluer(x)
        return x


class P2DecoderCrossAttention(nn.Module):
    '''
        The cross attention layer of the decoder allows the decoder to attend to the encoder output, that is add contextual about the image to the text.
    '''
    def __init__(self, embedding_dim: int, num_heads: int):
        super(P2DecoderCrossAttention, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

    def forward(self, decoder_self_attention_output: torch.Tensor, encoder_output: torch.Tensor) -> torch.Tensor:
        x, _ = self.cross_attn(decoder_self_attention_output, encoder_output, encoder_output) # the second parameter returned is the attention weights, we don't need them, for now anyways, and hopefully never
        return x
class P2GPTBlock(GPT2Model):
    '''
        The GPT block of the decoder is very similar to a GPT-2, with addition of a cross attention layer and decoupled embedding/de-embedding MLPs,
        this is done to allow fine-tuning of the model without touching the self-attention weights.
    '''
    def __init__(self, config: GPT2Config):
        super(P2GPTBlock, self).__init__(config=config)

    # taken straight from the transformers library
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        '''
        Args:
            input_ids: The input tensor of shape (batch_size, seq_length)
        Returns:
            A tensor of shape (batch_size, seq_length, d_model) containing the embeddings
        '''
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class P2Decoder(nn.Module):
    def __init__(self, gpt2_config: GPT2Config):
        super(P2Decoder, self).__init__()
        self.gpt2 = P2GPTBlock(gpt2_config)
        self.hidden_size = gpt2_config.n_embd
        self.vocab_size = gpt2_config.vocab_size
        self.cross_attention = P2DecoderCrossAttention(self.hidden_size, gpt2_config.n_head)# use the same number of heads as the GPT-2 model
        self.cross_attention_norm = DyT(self.hidden_size)
        # these are the embeddings that that the decoder outputs, the original GPT-2 model uses the same embeddings for input and output
        # but then we can't fine-tune the model without touching the self-attention weights
        # so we use a separate embedding layer for the output
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.lm_head.weight = nn.Parameter(self.gpt2.wte.weight.clone())

    def forward(self, x, encoder_output, attention_mask: Optional[torch.FloatTensor] = None):
        # batch_size, seq_length, d_model = gpt2_output.shape
        x_shape = x.shape
        encoder_output_shape = encoder_output.shape
        assert len(x_shape) == 2, f"Input shape mismatch: {x_shape}, format should be (batch_size, seq_length)"
        assert len(encoder_output_shape) == 3, f"Encoder output shape mismatch: {encoder_output_shape}, format should be (batch_size, seq_length, d_model)"
        assert x_shape[0] == encoder_output_shape[0], f"Batch size mismatch: {x_shape[0]} != {encoder_output_shape[0]}"
        logger.trace("Decoder input shape: {}", x.shape)
        logger.trace("Encoder output shape: {}", encoder_output.shape)
        x = self.gpt2(x, attention_mask=attention_mask).last_hidden_state # the output of the GPT-2 block is (batch_size, seq_length, d_model)
        logger.trace("Decoder output shape: {}", x.shape)
        x = self.cross_attention(x, encoder_output) + x
        x = self.cross_attention_norm(x)
        logger.trace("Cross attention output shape: {}", x.shape)
        x = self.lm_head(x)
        logger.trace("LM head output shape: {}", x.shape)
        return x

class P2ECDEC(nn.Module):
    def __init__(self, input_channels: int, input_width: int, input_height: int, d_model: int, decoder: nn.Module):
        super(P2ECDEC, self).__init__()
        self.encoder = P2Encoder(input_channels, input_width, input_height, d_model)
        self.decoder = decoder

    def forward(self, tokens: torch.Tensor, images: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        x = self.encoder(images)
        x = self.decoder(tokens, x, attention_mask=attention_mask)
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

if __name__ == "__main__":
    # Example usage
    encoder = P2Encoder(input_channels=3, input_width=224, input_height=224, d_model=768)
    # Create a random input tensor
    x = torch.randn(32, 3, 24, 224)  # (batch_size, channels, height, width)

    # Forward pass through the encoder
    encoded_x = encoder(x)
    print("Encoded shape:", encoded_x.shape)