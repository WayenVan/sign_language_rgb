from einops import rearrange
from torch import nn
from .lithrnet.build import build_litehrnet
from .lithrnet.litehrnet import IterativeHeadDownSample

class STHrnet(nn.Module):
    
    def __init__(
        self,
        hr_checkpoint,
        freeze_hrnet=True,
        ) -> None:
        super().__init__()
        
        self.freeze = freeze_hrnet
        
        #should be a lr net without header
        self.lrnet = build_litehrnet(hr_checkpoint)
        assert not hasattr(self.lrnet, 'head_layer')
        
        self.temporal_pool = nn.AvgPool3d((2, 1, 1), (2, 1, 1))

        self.header = IterativeHeadDownSample(self.lrnet.stages_spec['num_channels'][-1], self.lrnet.conv_cfg, self.lrnet.norm_cfg)
        self.outchannel = self.lrnet.stages_spec['num_channels'][-1][-1]
        #direct average pooling is better
        self.poolandflat = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(-3)
        )
        # self.forward_conv = nn.Sequential(
        #     nn.Conv2d(outchannel, d_model, 3, padding=(1, 1)),
        #     nn.BatchNorm2d(d_model),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1,1)),
        #     nn.Flatten(-3))
    
    def freeze_lrnet(self):
        for param in self.lrnet.parameters():
            param.requires_grad = False
        self.lrnet.norm_eval = True
        
    def unfreeze_lrnet(self):
        for param in self.lrnet.parameters():
            param.requires_grad = True
        self.lrnet.norm_eval = False
    
    def forward(self, x, video_length):
        """
        :param x: [t, n, c, h, w]
        :param video_length: [n]
        """
        T, N, C, H, W = x.shape
        x = rearrange(x, 't n c h w -> (t n) c h w')    
        x = self.lrnet.stem(x)

        x = rearrange(x, '(t n) c h w -> n c t h w', n=N)
        # x = self.temporal_pool(x)
        # video_length = video_length // 2
        x = rearrange(x, 'n c t h w -> (t n) c h w')

        y_list = [x]
        for i in range(self.lrnet.num_stages):
            x_list = []
            transition = getattr(self.lrnet, 'transition{}'.format(i))
            for j in range(self.lrnet.stages_spec['num_branches'][i]):
                #j refer to the j^th branch in i^th stage
                if transition[j]:
                    #if transition j exist
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    #if transition j not exist
                    x_list.append(y_list[j])
            y_list = getattr(self.lrnet, 'stage{}'.format(i))(x_list)

        #x: list([(tn) channel h w])
        x = y_list
    
        x = self.header(x)[-1]
        x = self.poolandflat(x)
        x = rearrange(x, '(t n) c -> t n c', n=N)
        #x [t n c]
        return x, video_length
    
    def train(self, mode: bool = True): 
        super().train(mode)
        if self.freeze:
            self.freeze_lrnet()
        else:
            self.unfreeze_lrnet()