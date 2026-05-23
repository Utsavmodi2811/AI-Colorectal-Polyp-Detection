import torch
import torch.nn as nn

# =====================================================
# BASIC CONV BLOCK
# =====================================================

def conv(in_channels, out_channels):

    return nn.Sequential(

        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(inplace=True),

        nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        ),

        nn.ReLU(inplace=True)
    )

# =====================================================
# U-NET
# =====================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.e1 = conv(3, 64)

        self.e2 = conv(64, 128)

        self.e3 = conv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.b = conv(256, 512)

        self.u2 = nn.ConvTranspose2d(
            512,
            256,
            2,
            2
        )

        self.d2 = conv(512, 256)

        self.u1 = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )

        self.d1 = conv(256, 128)

        self.u0 = nn.ConvTranspose2d(
            128,
            64,
            2,
            2
        )

        self.d0 = conv(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):

        e1 = self.e1(x)

        e2 = self.e2(self.pool(e1))

        e3 = self.e3(self.pool(e2))

        b = self.b(self.pool(e3))

        d2 = self.d2(
            torch.cat([self.u2(b), e3], 1)
        )

        d1 = self.d1(
            torch.cat([self.u1(d2), e2], 1)
        )

        d0 = self.d0(
            torch.cat([self.u0(d1), e1], 1)
        )

        return self.out(d0)

# =====================================================
# MINI U-NET
# =====================================================

class MiniUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                1,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):

        return self.net(x)

# =====================================================
# RES BLOCK
# =====================================================

class ResBlock(nn.Module):

    def __init__(self, in_c, out_c):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_c,
                out_c,
                3,
                1,
                1
            ),

            nn.ReLU(),

            nn.Conv2d(
                out_c,
                out_c,
                3,
                1,
                1
            )
        )

        self.skip = nn.Conv2d(
            in_c,
            out_c,
            1
        )

    def forward(self, x):

        return self.conv(x) + self.skip(x)

# =====================================================
# RES U-NET
# =====================================================

class ResUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.e1 = ResBlock(3, 64)

        self.e2 = ResBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.b = ResBlock(128, 256)

        self.up = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )

        self.out = nn.Conv2d(128, 1, 1)

    def forward(self, x):

        e1 = self.e1(x)

        e2 = self.e2(self.pool(e1))

        b = self.b(self.pool(e2))

        return self.out(self.up(b))

# =====================================================
# DEEP U-NET
# =====================================================

class DeepUNet(UNet):

    def __init__(self):

        super().__init__()

        self.extra = conv(512, 512)

    def forward(self, x):

        e1 = self.e1(x)

        e2 = self.e2(self.pool(e1))

        e3 = self.e3(self.pool(e2))

        b = self.b(self.pool(e3))

        b = self.extra(b)

        d2 = self.d2(
            torch.cat([self.u2(b), e3], 1)
        )

        d1 = self.d1(
            torch.cat([self.u1(d2), e2], 1)
        )

        d0 = self.d0(
            torch.cat([self.u0(d1), e1], 1)
        )

        return self.out(d0)

# =====================================================
# ATTENTION BLOCK
# =====================================================

class AttentionBlock(nn.Module):

    def __init__(self, Fg, Fl, Fi):

        super().__init__()

        self.Wg = nn.Conv2d(Fg, Fi, 1)

        self.Wx = nn.Conv2d(Fl, Fi, 1)

        self.psi = nn.Conv2d(Fi, 1, 1)

    def forward(self, g, x):

        return x * torch.sigmoid(
            self.psi(
                self.Wg(g) + self.Wx(x)
            )
        )

# =====================================================
# ATTENTION U-NET
# =====================================================

class AttentionUNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.e1 = conv(3, 64)

        self.e2 = conv(64, 128)

        self.pool = nn.MaxPool2d(2)

        self.b = conv(128, 256)

        self.up = nn.ConvTranspose2d(
            256,
            128,
            2,
            2
        )

        self.att = AttentionBlock(
            128,
            128,
            64
        )

        self.d = conv(256, 128)

        self.out = nn.Conv2d(128, 1, 1)

    def forward(self, x):

        e1 = self.e1(x)

        e2 = self.e2(self.pool(e1))

        b = self.b(self.pool(e2))

        d = self.up(b)

        e2 = self.att(d, e2)

        d = torch.cat([d, e2], 1)

        return self.out(self.d(d))