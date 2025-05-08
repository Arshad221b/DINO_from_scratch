import torch
import timm
from torch import nn

class DINOHead(nn.Module):
    def __init__(self, in_dim=384, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1)
        return self.last_layer(x)

# Create dummy data
B = 32  # Batch size
dummy_input = torch.randn(B, 3, 224, 224).cuda()

# Create model
vit = timm.create_model('vit_small_patch16_224', pretrained=True).cuda()
head = DINOHead(in_dim=vit.num_features).cuda()

# Profile forward pass
with torch.no_grad():
    features = vit.forward_features(dummy_input)  # [B, 197, 384]
    cls_token = features[:, 0]  # [B, 384]
    output = head(cls_token)    # [B, 65536]
    print(output.shape)