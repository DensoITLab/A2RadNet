import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers import to_2tuple
from .swin import SwinTransformer
from .angle_aware_encoder import AngleAwareEncoder
def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

class DetectionHead(nn.Module):

    def __init__(self, use_bn=True, reg_layer=2, input_angle_size=0):
        super(DetectionHead, self).__init__()
        self.use_bn = use_bn
        self.reg_layer = reg_layer
        self.input_angle_size = input_angle_size
        self.target_angle = 224
        bias = not use_bn
        if self.input_angle_size == 112:
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif self.input_angle_size == 224:
            self.conv1 = conv3x3(256, 144, bias=bias)
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif self.input_angle_size == 448:
            self.conv1 = conv3x3(256, 144, bias=bias, stride=(1, 2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias)
            self.bn2 = nn.BatchNorm2d(96)
        elif self.input_angle_size == 896:
            self.conv1 = conv3x3(256, 144, bias=bias, stride=(1, 2))
            self.bn1 = nn.BatchNorm2d(144)
            self.conv2 = conv3x3(144, 96, bias=bias, stride=(1, 2))
            self.bn2 = nn.BatchNorm2d(96)
        else:
            raise NameError('Wrong channel angle paraemter !')
            return
        self.conv3 = conv3x3(96, 96, bias=bias)
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = conv3x3(96, 96, bias=bias)
        self.bn4 = nn.BatchNorm2d(96)
        self.clshead = conv3x3(96, 1, bias=True)
        self.reghead = conv3x3(96, reg_layer, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.bn3(x)
        x = self.conv4(x)
        if self.use_bn:
            x = self.bn4(x)
        cls = torch.sigmoid(self.clshead(x))
        reg = self.reghead(x)
        return torch.cat([cls, reg], dim=1)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, residual=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual:
            out += x
        out = self.relu(out)
        return out

class RangePseudoAngleDecoder(nn.Module):

    def __init__(self, embed_dim=48, angle_feat_dim=112, patch_size=(2, 2), upsample_sizes=[(1, 2), (2, 2), (2, 2)], fpn_channels=128, out_channels=256, num_layers=4, basic_block_residual=False):
        super(RangePseudoAngleDecoder, self).__init__()
        assert num_layers == 4, 'This decoder is designed for 4 stages of features from the Swin Transformer backbone.'
        self.angle_feat_dim = angle_feat_dim
        self.patch_size = to_2tuple(patch_size)
        self.upsample_sizes = upsample_sizes
        self.num_layers = num_layers
        input_feat_dims = [embed_dim * 2 ** i for i in range(num_layers)]
        self.lat2 = nn.Conv2d(input_feat_dims[1], fpn_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(input_feat_dims[2], fpn_channels, kernel_size=1)
        self.lat4 = nn.Conv2d(input_feat_dims[3], fpn_channels, kernel_size=1)
        self.fuse43 = BasicBlock(fpn_channels * 2, fpn_channels, residual=basic_block_residual)
        self.fuse32 = BasicBlock(fpn_channels * 2, fpn_channels, residual=basic_block_residual)
        self.refine = BasicBlock(fpn_channels, out_channels, residual=basic_block_residual)

    def forward(self, features):
        F4 = self.lat4(features[3])
        F3 = self.lat3(features[2])
        F2 = self.lat2(features[1])
        X = F.interpolate(F4, scale_factor=self.upsample_sizes[2], mode='bilinear', align_corners=False)
        X = self.fuse43(torch.cat((X, F3), axis=1))
        X = F.interpolate(X, scale_factor=self.upsample_sizes[1], mode='bilinear', align_corners=False)
        X = self.fuse32(torch.cat((X, F2), axis=1))
        X = F.interpolate(X, scale_factor=self.upsample_sizes[0], mode='bilinear', align_corners=False)
        X = self.refine(X)
        return X

class A2RadNet(nn.Module):

    def __init__(self, patch_size, channels, in_chans, embed_dim, depths, num_heads, drop_rates, window_size=7, regression_layer=2, detection_head=True, segmentation_head=True, doppler_period=16, use_periodic_attention=None, range_window_size=1, merge_sizes=None, upsample_sizes=None, frontend_type=None, frontend_angle_estimator_type=1, frontend_embed_dim=32, frontend_out_channels=32, frontend_num_blocks=2, frontend_dropout=0.1, frontend_doppler_reducer=False, frontend_num_latents=8, frontend_use_cyclic_shift=True, freespace_upsample='bilinear', label_map={'detection': [128, 112], 'segmentation': [256, 224]}, basic_block_residual=False):
        super(A2RadNet, self).__init__()
        self.detection_head = detection_head
        self.segmentation_head = segmentation_head
        self.frontend = AngleAwareEncoder(doppler_bins=256, doppler_period=doppler_period, num_ant=16, complex_dim=2, embed_dim=frontend_embed_dim, out_channels=frontend_out_channels, num_latents=frontend_num_latents, num_blocks=frontend_num_blocks, dropout=frontend_dropout, doppler_reducer=frontend_doppler_reducer, angle_estimator_type=frontend_angle_estimator_type, use_cyclic_shift=frontend_use_cyclic_shift, output_order='ra')
        out_doppler_channels = frontend_num_latents if frontend_doppler_reducer else 256
        self.vit = SwinTransformer(pretrain_img_size=None, patch_size=patch_size, in_chans=out_doppler_channels, embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, mlp_ratio=4.0, qkv_bias=True, qk_scale=None, drop_rate=drop_rates[0], attn_drop_rate=drop_rates[1], drop_path_rate=drop_rates[2], norm_layer=nn.LayerNorm, ape=False, patch_norm=True, out_indices=(0, 1, 2, 3), frozen_stages=-1, use_checkpoint=False, doppler_period=doppler_period, use_periodic_attention=use_periodic_attention, range_window_size=range_window_size, merge_sizes=merge_sizes)
        self.RA_decoder = RangePseudoAngleDecoder(embed_dim=embed_dim, angle_feat_dim=frontend_out_channels, patch_size=patch_size, upsample_sizes=upsample_sizes if upsample_sizes is not None else [(2, 2), (2, 2), (2, 2), (2, 2)], fpn_channels=128, out_channels=256, basic_block_residual=basic_block_residual)
        if self.detection_head:
            self.detection_header = DetectionHead(input_angle_size=label_map['detection'][1], reg_layer=regression_layer)
        if self.segmentation_head:
            scale_factor = (label_map['segmentation'][0] // label_map['detection'][0], label_map['segmentation'][1] // label_map['detection'][1])
            self.freespace = nn.Sequential(nn.ConvTranspose2d(256, 256, kernel_size=3, stride=scale_factor, padding=1, output_padding=(1, 1)) if freespace_upsample == 'tconv' else nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False), BasicBlock(256, 128), BasicBlock(128, 64), nn.Conv2d(64, 1, kernel_size=1))

    def forward(self, x, return_features=False):
        """
        Input:
            x: [B, C, R, D] radar tensor after frontend processing (if frontend is used)
        Output:         dict with keys:
            'Detection': [B, 1+2, R, D] detection output
            'Segmentation': [B, 1, R', D'] segmentation output (R', D' depend on upsampling in decoder)
        """
        out = {'Detection': [], 'Segmentation': []}
        if self.frontend is not None:
            x = self.frontend(x)
        features = self.vit(x)
        RA = self.RA_decoder(features)
        if return_features:
            out['features'] = features
            out['RA'] = RA
        if self.detection_head:
            out['Detection'] = self.detection_header(RA)
        if self.segmentation_head:
            out['Segmentation'] = self.freespace(RA)
        return out
