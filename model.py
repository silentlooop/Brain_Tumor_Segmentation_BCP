import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    """(conv 3x3 -> BatchNorm -> ReLU) x 2. padding=1 keeps H and W unchanged."""

    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )


class UNet2D(nn.Module):
    """2D U-Net. Input (B, in_channels, H, W) -> logits (B, out_channels, H, W).

    Fully convolutional (no Linear layers), so any H, W divisible by 2**depth works.
    Each encoder level halves H, W and doubles the filters; the decoder mirrors it and
    concatenates the matching encoder output (skip connection) to recover fine detail.
    """

    def __init__(self, in_channels=4, out_channels=1, base_filters=32, depth=4):
        super().__init__()
        filters = [base_filters * 2 ** i for i in range(depth)]  # e.g. 32, 64, 128, 256

        self.downs = nn.ModuleList()
        c = in_channels
        for f in filters:
            self.downs.append(DoubleConv(c, f))
            c = f
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(filters[-1], filters[-1] * 2)

        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for f in reversed(filters):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.up_convs.append(DoubleConv(f * 2, f))  # f from upsampling + f from the skip

        self.head = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x, debug=False):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
            if debug:
                print(f"[model] down   {tuple(x.shape)}")

        x = self.bottleneck(x)
        if debug:
            print(f"[model] bottom {tuple(x.shape)}")

        for up, conv, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = conv(torch.cat([up(x), skip], dim=1))
            if debug:
                print(f"[model] up     {tuple(x.shape)}")

        return self.head(x)  # raw logits, no sigmoid (see loss_function.py)


def build_model(cfg):
    m = cfg["model"]
    return UNet2D(m["in_channels"], m["out_channels"], m["base_filters"], m["depth"])


if __name__ == "__main__":
    # Shape test: python model.py
    crop = 192
    net = UNet2D(4, 1, 32, 4)
    out = net(torch.randn(2, 4, crop, crop), debug=True)
    assert out.shape == (2, 1, crop, crop), out.shape
    print(f"OK: (2, 4, {crop}, {crop}) -> {tuple(out.shape)}, "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f}M parameters")
