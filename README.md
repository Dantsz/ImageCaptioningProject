# CNN-GPT2 Image captioning model

This project presents an image captioning model that integrates a custom convolutional neural network (CNN) encoder with a GPT2-based autoregressive decoder. The encoder, referred to as P3Encoder, is designed from scratch and incorporates Squeeze-and-Excitation (SE) layers and learned two-dimensional positional encodings to enhance spatial feature extraction and channel-wise attention. These encoded image features are fed into a modified GPT-2 decoder through cross-attention layers, enabling the model to generate contextually rich captions while preserving the sequential generation capabilities of the original transformer. I trained the model end-to-end on the MS-COCO dataset, comprising approximately 500,000 image-caption pairs. Optimization is performed using AdamW with cosine annealing learning rate scheduling. Experimental results demonstrate the model's ability to generate fluent and relevant captions, achieving a METEOR score of 29, which highlights the effectiveness of combining structured visual feature extraction with powerful language modeling. This approach avoids modifying GPT-2's self-attention mechanism, focusing instead on clean integration via cross-modal attention for scalable and modular deployment.

## Architecture

![Architecture](https://github.com/Dantsz/ImageCaptioningProject/blob/main/resources/architecture.png?raw=true)

## Comfyui nodes

Folder `comfyui_nodes` contains custom nodes for ComfyUI:

![ComfyUI Nodes](https://github.com/Dantsz/ImageCaptioningProject/blob/main/resources/comfyui_nodes.png?raw=true)
