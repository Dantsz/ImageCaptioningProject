import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

class BinaryViT(nn.Module):
    def __init__(self):
        super(BinaryViT, self).__init__()
        self.vit = vit_b_16()
        # replacing the head
        self.vit.heads = nn.Sequential(nn.Linear(self.vit.hidden_dim, 2))
    def forward(self, x):
        x = self.vit(x)
        return x

if __name__ == '__main__':
    # Example Usage:
    model = BinaryViT()
    print("Created the binary ViT!")
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # Output shape: [1, 1]

    #Loss function.
    criterion = nn.BCEWithLogitsLoss()

    #Example of calculating loss.
    target = torch.randn(1, 1) #example target.
    loss = criterion(output, target)
    print(loss)
