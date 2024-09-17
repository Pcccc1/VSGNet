from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.runner import (BaseModule, load_state_dict)
import math

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class FR(nn.Module):
    def __init__(self, dim):
        super(FR, self).__init__()
        self.c1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.c2 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.MLP_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.MLP_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1), nn.Conv2d(dim, dim, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()
        self.proj1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.proj2 = nn.Conv2d(dim * 2, dim, kernel_size=1)

    def forward(self, x1, x2, FE_x1, FE_x2):
        N, C, H, W = x1.size()
        xc = torch.cat([x1, x2], dim=1)
        x1x = self.c1(xc)
        x2x = self.c2(xc)

        x1_exp = x1x.exp()
        softmax_x1 = F.softmax(x1_exp.view(N, C, -1), dim=2).view_as(x1x)
        we_x1 = softmax_x1 * x1x
        sm_x1 = we_x1.sum(dim=(2, 3), keepdim=True)
        fg_x1 = self.sigmoid(self.MLP_1(sm_x1))
        weight_x1 = x1 * fg_x1

        x2_exp = x2x.exp()
        softmax_x2 = F.softmax(x2_exp.view(N, C, -1), dim=2).view_as(x2x)
        we_x2 = softmax_x2 * x2x
        sm_x2 = we_x2.sum(dim=(2, 3), keepdim=True)
        fg_x2 = self.sigmoid(self.MLP_2(sm_x2))
        weight_x2 = x2 * fg_x2

        re_x1 = weight_x1 + x2
        re_x2 = weight_x2 + x1

        sp_x1 = re_x1.exp()
        softmax_x1_sp = F.softmax(sp_x1, dim=1)
        we_x1_sp = softmax_x1_sp * re_x1
        sm_x1_sp = we_x1_sp.sum(dim=1, keepdim=True)
        weight_x1_sp = x1 * sm_x1_sp

        sp_x2 = re_x2.exp()
        softmax_x2_sp = F.softmax(sp_x2, dim=1)
        we_x2_sp = softmax_x2_sp * re_x2
        sm_x2_sp = we_x2_sp.sum(dim=1, keepdim=True)
        weight_x2_sp = x2 * sm_x2_sp

        fg_x1_sp = weight_x1_sp + x2
        fg_x2_sp = weight_x2_sp + x1

        co_x1 = fg_x1_sp + FE_x1
        co_x2 = fg_x2_sp + FE_x2

        re = torch.cat([co_x1, co_x2], dim=1)

        po_x1 = self.proj1(re)
        po_x2 = self.proj2(re)

        return po_x1, po_x2
    

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, norm_cfg=dict(type='SyncBN', requires_grad=True)):
        super().__init__()

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio, 3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pos(x) + x
        x = x.permute(0, 2, 3, 1)
        x = self.act(x)
        x = self.fc2(x)

        return x


class FE(nn.Module):
    def __init__(self, dim, num_head=8, norm_cfg=dict(type='SyncBN', requires_grad=True),drop_depth=False):
        super().__init__()
        self.num_head = num_head
        self.window = 7

        self.q = nn.Linear(dim, dim)
        self.q_cut = nn.Linear(dim, dim//2)
        self.a = nn.Linear(dim, dim)
        self.l = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.e_conv = nn.Conv2d(dim//2, dim//2, 7, padding=3, groups=dim//2)
        self.e_fore = nn.Linear(dim, dim//2)
        self.e_back = nn.Linear(dim//2, dim//2)

        self.proj = nn.Linear(dim*2, dim)
        if not drop_depth:
            self.proj_e = nn.Linear(dim*2, dim//2)

        self.short_cut_linear = nn.Linear(2*dim, dim//2)
        self.kv = nn.Linear(dim, dim)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.proj = nn.Linear(dim, dim)
        if not drop_depth:
            self.proj_e = nn.Linear(dim, dim)

        self.act = nn.GELU()
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.norm_e = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.drop_depth = drop_depth 

    def forward(self, x,x_e):
        B, H, W, C = x.size()
        x = self.norm(x)
        x_e = self.norm_e(x_e)
        short_cut = torch.cat([x,x_e],dim=3)
        short_cut = short_cut.permute(0,3,1,2)

        cutted_x = self.q_cut(x)        
        x = self.l(x).permute(0, 3, 1, 2)
        x = self.act(x)
            
        b = x.permute(0, 2, 3, 1)
        kv = self.kv(b)
        kv = kv.reshape(B, H*W, 2, self.num_head, C // self.num_head // 2).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        short_cut = self.pool(short_cut).permute(0,2,3,1)
        short_cut = self.short_cut_linear(short_cut)
        short_cut = short_cut.reshape(B, -1, self.num_head, C // self.num_head // 2).permute(0, 2, 1, 3)
        m = short_cut
        attn = (m * (C // self.num_head // 2) ** -0.5) @ k.transpose(-2, -1) 
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).reshape(B, self.num_head, self.window, self.window, C // self.num_head // 2).permute(0, 1, 4, 2, 3).reshape(B, C // 2, self.window, self.window)
        attn = F.interpolate(attn, (H, W), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
    
        x_e = self.e_back(self.e_conv(self.e_fore(x_e).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))
        cutted_x = cutted_x * x_e


        x = torch.cat([attn,cutted_x], dim=3)

        if not self.drop_depth:
            x_e = self.proj_e(x)
        x = self.proj(x)

        return x,x_e

class Block(nn.Module):
    def __init__(self, index, dim, num_head, norm_cfg=dict(type='SyncBN', requires_grad=True), mlp_ratio=4., dropout_layer=None,drop_depth=False):
        super().__init__()
        
        self.index = index
        layer_scale_init_value = 1e-6  

        self.attn = FE(dim, num_head, norm_cfg=norm_cfg,drop_depth=drop_depth)
        self.mlp = MLP(dim, mlp_ratio, norm_cfg=norm_cfg)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
                 
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        
        if not drop_depth:
            self.layer_scale_1_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
            self.layer_scale_2_e = nn.Parameter(layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
            self.mlp_e2 = MLP(dim//2, mlp_ratio)
        self.drop_depth=drop_depth

    def forward(self, x, x_e):
        res_x,res_e=x,x_e
        x,x_e=self.attn(x,x_e)

        
        x = res_x + self.dropout_layer(self.layer_scale_1.unsqueeze(0).unsqueeze(0) * x )

        
        x = x + self.dropout_layer(self.layer_scale_2.unsqueeze(0).unsqueeze(0) * self.mlp(x))
        if not self.drop_depth:
            x_e = res_e + self.dropout_layer(self.layer_scale_1_e.unsqueeze(0).unsqueeze(0) * x_e)
            x_e = x_e + self.dropout_layer(self.layer_scale_2_e.unsqueeze(0).unsqueeze(0) * self.mlp_e2(x_e))

        return x,x_e


class VSGNet(BaseModule):
    def __init__(self, depths=(2, 2, 8, 2), dims=(32, 64, 128, 256), out_indices=(0, 1, 2, 3), norm_cfg=dict(type='SyncBN', requires_grad=True),
                 mlp_ratios=[8, 8, 4, 4], num_heads=(2, 4, 10, 16),last_block=[50,50,50,50], drop_path_rate=0.1, init_cfg=None):
        super().__init__()
        print(drop_path_rate)
        self.depths = depths
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.backbone_rgb = torch.hub.load(
            "dinov2", "dinov2_vitg14_reg", source="local", pretrained=False
        )
        self.backbone_rgb.load_state_dict(torch.load("dinov2_vitg14_reg4_pretrain.pth"))
        self.backbone_e = torch.hub.load(
            "dinov2", "dinov2_vitb14_reg", source="local", pretrained=False
        )
        self.backbone_e.load_state_dict(torch.load("dinov2_vitb14_reg4_pretrain.pth"))
        self.downsample_layers = nn.ModuleList() 

        self.FRlist = nn.ModuleList()
        for i in range(4):
            self.FRlist.append(FR(dim=dims[i]))

        stem = nn.Sequential(
                nn.Conv2d(3, dims[0] // 2, kernel_size=3,stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
        )

        self.downsample_layers_e = nn.ModuleList() 
        stem_e = nn.Sequential(
                nn.Conv2d(1, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
        )

        self.dn = nn.Sequential(
                nn.Conv2d(24, dims[0] // 2, kernel_size=3,stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 2),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
        )
        self.dn_e = nn.Sequential(
                nn.Conv2d(12, dims[0] // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0] // 4),
                nn.GELU(),
                nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(dims[0]),
        )


        self.downsample_layers.append(stem)
        self.downsample_layers_e.append(stem_e)

        for i in range(len(dims)-1):
            stride = 2
            downsample_layer = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i])[1],
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers.append(downsample_layer)

            downsample_layer_e = nn.Sequential(
                    build_norm_layer(norm_cfg, dims[i])[1],
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer_e)

        self.fe = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):
            stage = nn.Sequential(
                *[Block(index=cur+j, 
                        dim=dims[i], 
                        dropout_layer=dict(type='DropPath', drop_prob=dp_rates[cur + j]), 
                        num_head=num_heads[i], 
                        norm_cfg=norm_cfg,
                        block_index=depths[i]-j,
                        last_block_index=last_block[i],
                        mlp_ratio=mlp_ratios[i],drop_depth=((i==3)&(j==depths[i]-1)),) for j in range(depths[i])],
            )
            self.fe.append(stage)
            cur += depths[i]

       
        for i in out_indices:
            layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)


    def init_weights(self,pretrained):
       
        _state_dict=torch.load(pretrained, map_location=torch.device('cpu'))
        if 'state_dict_ema' in _state_dict.keys():
            _state_dict=_state_dict['state_dict_ema']
        else:
            _state_dict=_state_dict['state_dict']

        state_dict = OrderedDict()
        for k, v in _state_dict.items():
            if k.startswith('backbone.'):
                state_dict[k[9:]] = v
            else:
                state_dict[k] = v

        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        
        load_state_dict(self, state_dict, strict=False)

    def forward(self, x,x_e):
        if x_e is None:
            x_e=x
        if len(x.shape)==3:
            x=x.unsqueeze(0)
        if len(x_e.shape)==3:
            x_e=x_e.unsqueeze(0)

        x_e=x_e[:,0,:,:].unsqueeze(1)

        for name, param in self.backbone_rgb.named_parameters():
            param.requires_grad = False
        for name, param in self.backbone_e.named_parameters():
            param.requires_grad = False        

        outs = []
        residual, residual_e = x, x_e
        sh = x.shape[2]
        sh_e = x_e.shape[2]
        sh = math.ceil(sh/14)*14
        sh_e = math.ceil(sh_e/14)*14
        x_e = torch.cat((x_e, x_e, x_e), dim=1)
        x = F.interpolate(x, size=(sh, sh), mode='bilinear', align_corners=False)
        x_e = F.interpolate(x_e, size=(sh_e, sh_e), mode='bilinear', align_corners=False)
        x = self.backbone_rgb.forward_features(x)["x_norm_patchtokens"]
        x_e = self.backbone_e.forward_features(x_e)["x_norm_patchtokens"]
        x = x.reshape(-1, 296, 296, 24)
        x_e = x_e.reshape(-1, 296, 296, 12)
        x = x.permute(0, 3, 1, 2)
        x_e = x_e.permute(0, 3, 1, 2)

        x = F.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)
        x_e = F.interpolate(x_e, size=(512, 512), mode='bilinear', align_corners=False)


        for i in range(4):
            if i == 0:
                residual = self.downsample_layers[i](residual)
                residual_e = self.downsample_layers_e[i](residual_e)
                x = self.dn(x)
                x_e = self.dn_e(x_e)

                x = x + residual
                x_e = x_e  + residual_e
            else:
                x = self.downsample_layers[i](x)
                x_e = self.downsample_layers_e[i](x_e)
           
            x = x.permute(0, 2, 3, 1)
            x_e = x_e.permute(0, 2, 3, 1)
            for i in range(4):
                hx, hx_e = self.fe[i](x,x_e)
                x, x_e = self.FRlist[i](x, x_e, hx, hx_e)
            x = x.permute(0, 3, 1, 2)
            x_e = x_e.permute(0, 3, 1, 2)
            outs.append(x)
        return outs




def VSGNet(pretrained=False,drop_path_rate=0.1, **kwargs): 
    model = VSGNet(dims=[96, 192, 288, 576], mlp_ratios=[8, 8, 4, 4], depths=[3, 3, 12, 2], num_heads=[1, 2, 4, 8], drop_path_rate=drop_path_rate, num_headss=[3, 6, 12 ,24], **kwargs)
    if pretrained:
        model = load_model_weights(model, 'scnet', kwargs)
    return model
