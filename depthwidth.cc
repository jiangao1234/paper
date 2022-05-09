def qConv(in_channels, out_channels, kernel_size=None, stride=1, pad=1, groups=1, dilation=1, bias=False, bn=True, act=False, first=False, last=False):
    assert(groups==1)
    assert(dilation==1)
    assert(out_channels % 32 == 0)
    act_fn = nn.ReLU6(inplace=True) if not IS_QUANTIZE and act else None
    if isinstance(kernel_size, tuple):
        h = int(kernel_size[0])
        w = int(kernel_size[1])
    else:
        h = w = int(kernel_size)
    return nn.Sequential(
            ops.DepthwiseConv2D(in_channels,
                        kernel_h = 3,
                        kernel_w = 3,
                        stride = stride,
                        activation_fn = act_fn,
                        enable_batch_norm = bn,
                        enable_bias = bias,
                        quantize = IS_QUANTIZE,
                        padding = pad,auto_bitwidth=False,
                        weight_bitwidth = BITW,
                        input_bitwidth = BITA,
                        output_bitwidth = BITA,
                        clip_max_value = CLIP_MAX_VALUE,
                        weight_factor = WEIGHT_FACTOR,
                        target_device = TARGET_DEVICE),
            ops.Conv2D(in_channels,
                        out_channels,
                        kernel_h = 1,
                        kernel_w = 1,
                        stride = 1,
                        activation_fn = act_fn,
                        enable_batch_norm = bn,
                        enable_bias = bias,auto_bitwidth=False,
                        quantize = IS_QUANTIZE,
                        first_layer = first,
                        padding = 0,
                        weight_bitwidth = BITW,
                        input_bitwidth = BITA,
                        output_bitwidth = BITA,
                        clip_max_value = CLIP_MAX_VALUE,
                        weight_factor = WEIGHT_FACTOR,
                        target_device = TARGET_DEVICE))
def add(channels):
    assert(qcfg['quan'] == IS_QUANTIZE)
    return ops.Shortcut(channels,
                        quantize = IS_QUANTIZE,
                        input_bitwidth = BITA,
                        output_bitwidth = BITA,
                        clip_max_value = CLIP_SHORTCUT,
                        target_device = TARGET_DEVICE)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1,pad=1,bias=False,bn=True,act=False, first=False, last=False):
    """3x3 convolution with padding"""
    #return qConv(in_planes, out_planes, kernel_size=3, stride=stride,
    #                 padding=dilation, groups=groups, bias=False, dilation=dilation)
    return qConv(in_planes, out_planes, kernel_size=3, stride=stride,
                 pad=pad, groups=groups, bias=False, dilation=dilation,bn=bn,act=act,first=first,last=last)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,bn=True,act=True)
        self.conv2 = conv3x3(planes, planes,bn=True,act=True)
        self.downsample = downsample
        self.add=add(planes)
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out=self.add((out,identity))

        return out


