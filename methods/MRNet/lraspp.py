from torch import nn

class LRASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
        self.aspp2 = nn.Sequential(
            #nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.asppr4 = nn.Sequential(
            #nn.Upsample( scale_factor=4, mode='nearest', align_corners=None),
            nn.Conv2d(128, 64, 1, bias=False)

        )
        self.asppr3 = nn.Sequential(
            nn.Upsample( scale_factor=2, mode='nearest', align_corners=None),
            #nn.Conv2d(out_channels, 1, 1, bias=False),
            nn.Conv2d(128, 40, 1, bias=False)

        )
        self.asppr2 = nn.Sequential(
            nn.Upsample( scale_factor=4, mode='nearest', align_corners=None),
            #nn.Conv2d(out_channels, 1, 1, bias=False),
            nn.Conv2d(128, 20, 1, bias=False)

        )
        self.asppr1 = nn.Sequential(
            nn.Upsample( scale_factor=8, mode='nearest', align_corners=None),
            #nn.Conv2d(out_channels, 1, 1, bias=False),
            nn.Conv2d(128, 16, 1, bias=False)

        )

    def forward_single_frame(self, x1,x2):
        x2 =  x1+x2
        print(f'plus {x2.shape}')
        x2 = self.aspp2(x2)
        print(f'x2 {x2.shape}')
        x_t = self.aspp2(x1)
        print(f'x_t {x_t.shape}')
        # print(f'x_t{x_t.shape}')
        return self.aspp1(x1) *x2, self.asppr4(x2), self.asppr3(x2),self.asppr2(x_t),self.asppr1(x_t)

    def forward_time_series(self, x):
        B, T = x.shape[:2]
        x = self.forward_single_frame(x.flatten(0, 1)).unflatten(0, (B, T))
        return x

    def forward(self, x1,x2):

        return self.forward_single_frame(x1,x2)
