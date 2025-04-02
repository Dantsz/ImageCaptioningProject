import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer
from PIL import Image
from loguru import logger

class TransformerDecoderWithCrossAttention(nn.Module):
    def __init__(self, embedding_dim, vocab_size, max_length, gpt2_model_name='gpt2'):
        super(TransformerDecoderWithCrossAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Load GPT-2 configuration
        gpt2_config = GPT2Config.from_pretrained(gpt2_model_name)
        self.hidden_dim = gpt2_config.n_embd
        self.num_heads = gpt2_config.n_head

        # Linear layer to project image embeddings into GPT-2 embedding space
        self.image_projection = nn.Linear(embedding_dim, self.hidden_dim)

        # Load pre-trained GPT-2 model
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

        # Freeze GPT-2 parameters,Ideally we would use stock GPT2 and freeze the weights, but we use the head and attention after cross-attention
        # and freezing wouldn't prop the grads to the encoder and we'd lose a lot of money(more than I lost already by having this uncommented so far :()
        #for param in self.gpt2.parameters():
        #    param.requires_grad = False


        assert len(self.gpt2.transformer.h) > 0, "No heads in the GPT2?"
        # Add cross-attention layers to each GPT-2 transformer block
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(self.hidden_dim, self.num_heads, batch_first=True)
            for _ in self.gpt2.transformer.h
        ])

        # Learnable positional embeddings for image input
        self.image_positional_embedding = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

    def forward(self, image_embeddings, target_tokens=None, attention_mask=None):
        batch_size = image_embeddings.size(0)

        # Project image embeddings into GPT-2 embedding space
        projected_image_embeddings = self.image_projection(image_embeddings)# + self.image_positional_embedding #for some reason this fucks everything up

        if target_tokens is not None:
            # Prepare input tokens (prepend BOS token)
            start_tokens = torch.ones((batch_size, 1), dtype=torch.long, device=image_embeddings.device) * self.gpt2.config.bos_token_id
            input_tokens = torch.cat([start_tokens, target_tokens], dim=1)
            input_embeddings = self.gpt2.transformer.wte(input_tokens)

            # Create attention mask (1 for valid tokens, 0 for padding)
            if attention_mask is None:
                pad_token_id = self.gpt2.config.pad_token_id or 0  # Default to 0 if pad_token_id is None
                attention_mask = (input_tokens != pad_token_id).long()

            # Create causal mask to prevent future token access
            seq_length = input_tokens.size(1)
            causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=image_embeddings.device)).unsqueeze(0).unsqueeze(0)

            # Stack all hidden states and perform self-attention in parallel
            hidden_states = torch.stack([layer.attn(hidden_states=input_embeddings, attention_mask=causal_mask)[0]
                                        for layer in self.gpt2.transformer.h], dim=0)

            # Stack the cross-attention outputs in parallel
            cross_attn_outputs = torch.stack([self.cross_attention[idx](hidden_states[idx], projected_image_embeddings, projected_image_embeddings)[0]
                                              for idx in range(hidden_states.shape[0])], dim=0)

            # Combine all the results (residual connections)
            input_embeddings = input_embeddings + sum(cross_attn_outputs)
            input_embeddings = input_embeddings.contiguous()

            # Get final logits
            outputs = self.gpt2(inputs_embeds=input_embeddings, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits[:, :-1, :]
            return logits
        else:
          generated_tokens = torch.ones((batch_size, 1), dtype=torch.long, device=image_embeddings.device) * self.gpt2.config.bos_token_id
          all_generated_tokens = generated_tokens
          input_embeddings = self.gpt2.transformer.wte(generated_tokens)

          for _ in range(self.max_length):
              seq_length = input_embeddings.size(1)
              causal_mask = torch.tril(torch.ones(seq_length, seq_length, device=image_embeddings.device)).unsqueeze(0).unsqueeze(0)

              # Stack all hidden states and perform self-attention in parallel
              hidden_states = torch.stack([layer.attn(hidden_states=input_embeddings, attention_mask=causal_mask)[0]
                                          for layer in self.gpt2.transformer.h], dim=0)

              # Stack the cross-attention outputs in parallel
              cross_attn_outputs = torch.stack([self.cross_attention[idx](hidden_states[idx], projected_image_embeddings, projected_image_embeddings)[0]
                                                for idx in range(hidden_states.shape[0])], dim=0)
              # Combine all the results (residual connections)
              input_embeddings = input_embeddings + sum(cross_attn_outputs)
              input_embeddings = input_embeddings.contiguous()

              # Get logits for the last token
              outputs = self.gpt2(inputs_embeds=input_embeddings, use_cache=False)
              logits = outputs.logits[:, -1, :]

              # Predict next token (you might want to change this to sampling for diversity)
              predicted_token = torch.argmax(logits, dim=-1).unsqueeze(1)

              # Append the predicted token to the sequence
              all_generated_tokens = torch.cat([all_generated_tokens, predicted_token], dim=1)

              # Check if the <EOS> token is generated and break if so
              if predicted_token.item() == self.gpt2.config.eos_token_id:
                  break

              # Update input embeddings with the new token
              input_embeddings = self.gpt2.transformer.wte(predicted_token)

          return all_generated_tokens[:, 1:]  # Exclude the initial <BOS> token

class HandmadeEncoderDecoder(nn.Module):
    def __init__(self):
        super(HandmadeEncoderDecoder, self).__init__()
        ENCODER_CHEKCPOINT = "google/vit-base-patch16-224-in21k" # use the pre-trained ViT model for now
        self.encoder = ViTModel.from_pretrained(ENCODER_CHEKCPOINT)
        self.embedding_dim = self.encoder.config.hidden_size # the hidden size of the encoder is the size of the image embeddings generated, without any specific head, it should be a representation of the image
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")# I don't know much about the tokenizer, but we need it here for the vocab size, so the decoder knows the lenth of the *thing* which returns the probability of then next token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.max_length = 20 # the maximum length of the caption, this is a hyperparameter, we can change it later
        self.decoder = TransformerDecoderWithCrossAttention(self.embedding_dim, self.vocab_size, self.max_length)
        logger.trace("HandmadeEncoderDecoder initialized with embedding_dim: {}, vocab_size: {}, max_length: {}", self.embedding_dim, self.vocab_size, self.max_length)

    def forward(self, image: Image.Image, target_tokens=None):
        '''
            target_tokens - used for training, represents the caption so far and because the model is autoregressive, we need to pass the previous tokens to get the next token
            image - the image to be encoded, shape (batch_size, 3, 224, 224)
        '''
        batch_size = image.size(0)
        encoder_output = self.encoder(image).last_hidden_state.mean(dim=1).unsqueeze(1)
        #print(f'image shape: {image.shape}, encoder_output shape: {encoder_output.shape}, target_tokens shape: {target_tokens.shape if target_tokens is not None else None}')
        decoder_output = self.decoder(encoder_output, target_tokens)
        if target_tokens is not None:
            return decoder_output
        else:
            return self.tokenizer.batch_decode(decoder_output.cpu().tolist(), skip_special_tokens=True)