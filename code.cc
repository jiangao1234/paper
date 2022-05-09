附录一 CRNN 模型具体实现代码
import torch
import torch.nn.functional as F
class Vgg_16(torch.nn.Module):
    def __init__(self):
        super(Vgg_16, self).__init__()
        self.convolution1 = torch.nn.Conv2d(1, 64, 3, padding=1)
        self.pooling1 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution2 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.pooling2 = torch.nn.MaxPool2d(2, stride=2)
        self.convolution3 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.convolution4 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.pooling3 = torch.nn.MaxPool2d((1, 2), stride=(2, 1)) # notice stride of the non-square pooling
        self.convolution5 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.BatchNorm1 = torch.nn.BatchNorm2d(512)
        self.convolution6 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.BatchNorm2 = torch.nn.BatchNorm2d(512)
        self.pooling4 = torch.nn.MaxPool2d((1, 2), stride=(2, 1))
        self.convolution7 = torch.nn.Conv2d(512, 512, 2)
    def forward(self, x):
        x = F.relu(self.convolution1(x), inplace=True)
        x = self.pooling1(x)
        x = F.relu(self.convolution2(x), inplace=True)
        x = self.pooling2(x)
        x = F.relu(self.convolution3(x), inplace=True)
        x = F.relu(self.convolution4(x), inplace=True)
        x = self.pooling3(x)
        x = self.convolution5(x)
        x = F.relu(self.BatchNorm1(x), inplace=True)
        x = self.convolution6(x)
        x = F.relu(self.BatchNorm2(x), inplace=True)
        x = self.pooling4(x)
        x = F.relu(self.convolution7(x), inplace=True)
        return x  # b*512x1x16
class RNN(torch.nn.Module):
    def __init__(self, class_num, hidden_unit):
        super(RNN, self).__init__()
        self.Bidirectional_LSTM1 = torch.nn.LSTM(512, hidden_unit, bidirectional=True)
        self.embedding1 = torch.nn.Linear(hidden_unit * 2, 512)
        self.Bidirectional_LSTM2 = torch.nn.LSTM(512, hidden_unit, bidirectional=True)
        self.embedding2 = torch.nn.Linear(hidden_unit * 2, class_num)
    def forward(self, x):
        x = self.Bidirectional_LSTM1(x)   # LSTM output: output, (h_n, c_n)
        T, b, h = x[0].size()   # x[0]: (seq_len, batch, num_directions * hidden_size)
        x = self.embedding1(x[0].view(T * b, h))  # pytorch view() reshape as [T * b, nOut]
        x = x.view(T, b, -1)  # [16, b, 512]
        x = self.Bidirectional_LSTM2(x)
        T, b, h = x[0].size()
        x = self.embedding2(x[0].view(T * b, h))
        x = x.view(T, b, -1)
        return x  # [16,b,class_num]
# output: [s,b,class_num]
class CRNN(torch.nn.Module):
    def __init__(self, class_num, hidden_unit=256):
        super(CRNN, self).__init__()
        self.cnn = torch.nn.Sequential()
        self.cnn.add_module('vgg_16', Vgg_16())
        self.rnn = torch.nn.Sequential()
        self.rnn.add_module('rnn', RNN(class_num, hidden_unit))
    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        # print(x.size()): b,c,h,w
        assert h == 1   # "the height of conv must be 1"
        x = x.squeeze(2)  # remove h dimension, b *512 * width
        x = x.permute(2, 0, 1)  # [w, b, c] = [seq_len, batch, input_size]
        # x = x.transpose(0, 2)
        # x = x.transpose(1, 2)
        x = self.rnn(x)
        return x

附录二 深度可分离卷积代替普通卷积参考代码
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

附录三 cat和add示例代码
  cat实例：
import torch
a = torch.ones([1, 2])
b = torch.ones([1, 2])
c = torch.cat([a, b], 1)
print(c)
print(c.type)
 add实例：
x = torch.rand(5, 3)
y = torch.rand(5, 3)
print(x)
print(y)
#method 1
print(x + y)
#method 2
print(torch.add(x, y))
附录四 simd指令优化代码
优化前代码：
static void forward(int8_t* input_data, int8_t* filter_data,
		int32_t* bias_data, int8_t* output_data0,
		int32_t* mul_data_pos, int32_t* shift_data_pos, 
		int input_depth, int output_depth,
		int input_height, int input_width,
		int output_height, int output_width,
		int kernel_height, int kernel_width, int* strides)
{
	int8_t* in_data = input_data;
	int8_t* ft_data = filter_data;
	int8_t* output_data = output_data0;
	int32_t* bias = bias_data;
	int32_t* mul_pos=mul_data_pos;
	int32_t* shift_pos=shift_data_pos;
	int32_t* out_char_ptr;

	int8_t* out_ptr;
	int8_t* in_ptr;
	int8_t* in_loc;
	int8_t* in_locr;
	int8_t* ft_ptr;
	int8_t* ft_loc;
	int8_t* ft_locr;
	int in_h;
	int in_w;
	int out_w;
	int out_h;
	int eight_bit_res;
	int sum1;

	int filter_count = output_depth;

	for(int i = 0; i < output_width * output_height; ++i)
	{
		out_h = i / output_width;
		out_w = i % output_width;
		out_ptr = output_data + i * output_depth;
		in_h = out_h * strides[0];
		in_w = out_w * strides[1];

		ft_ptr = ft_data;
		in_ptr = in_data + ((in_h) * input_width + (in_w)) * input_depth;

		for(int j = 0; j < output_depth; ++j)
		{
			in_loc = in_ptr + j;
			ft_loc = ft_ptr + j;
			sum1 = 0;
			for(int fi = 0; fi < kernel_height; ++fi)
			{
				int inh = in_h + fi;
				in_locr = in_loc + fi * (input_width*input_depth);
				ft_locr = ft_loc + fi * (kernel_width*filter_count);
				for(int fj = 0; fj < kernel_width; ++fj)
				{
					sum1 += ft_locr[fj*filter_count] * in_locr[fj*input_depth];
				}
			}
			sum1 += bias[j];
            out_ptr[j] = quantized_src(sum1, mul_pos[j], shift_pos[j], false);
            //if(sum1 < 0)
            //{
            //    out_ptr[j] = -128;
            //    continue;
            //}
            //eight_bit_res = MultiplyByQuantizedMultiplierSmallerThanOne_new(sum1, mul_pos[j], shift_pos[j], false);
            //eight_bit_res = std::min(eight_bit_res, 255);
            //out_ptr[j] = eight_bit_res - 128;
		}
	}
}
使用simd指令优化后代码：
#include "forward.h"
#include "common.h"
#include "assert.h"
#include <stdio.h>
#include <msa.h>
#include <limits>
#include <cmath>

static int32_t max_precision = 31;
static int action_min_fb = 0;
static int action_max_fb = 255;
static int action_min_hb = 0;
static int action_max_hb = 15;
//static int32_t max_precision = 31;
//static int action_min_fb = -128;
//static int action_max_fb = 127;
//static int action_min_hb = -8;
//static int action_max_hb = 7;

static int32_t Dup(int32_t x) {
    return x;
}

static int32_t BitAnd(int32_t a, int32_t b) {
    return a & b;
}

static int32_t BitNot(int32_t a) {
    return ~a;
}

static int32_t Add(int32_t a, int32_t b) {
    return a + b;
}

static int32_t ShiftRight(int32_t a, int offset) {
    return a >> offset;
}

static int32_t MaskIfNonZero(int32_t a) {
    static const int32_t zero = 0;
    return a ? BitNot(zero) : zero;
}

static int32_t MaskIfLessThan(int32_t a, int32_t b) {
    return MaskIfNonZero(a < b);
}

static int32_t MaskIfGreaterThan(int32_t a, int32_t b) {
    return MaskIfNonZero(a > b);
}

static int32_t RoundingDivideByPOT(int32_t x, int exponent, bool p) {
    assert(exponent >= 0);
    assert(exponent <= max_precision);
    const int32_t mask = Dup((1ll << exponent) - 1);
    const int32_t zero = Dup(0);
    const int32_t one = Dup(1);
    const int32_t remainder = BitAnd(x, mask);
    const int32_t threshold = Add(ShiftRight(mask, 1),
            BitAnd(MaskIfLessThan(x, zero), one));

    return Add(ShiftRight(x, exponent),
            BitAnd(MaskIfGreaterThan(remainder, threshold), one));
}

static int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b, bool p) {

    bool overflow = a == b && a == std::numeric_limits < int32_t > ::min();
    int64_t a_64 = a;
    int64_t b_64 = b;
    int64_t ab_64 = a_64 * b_64;
    int32_t nudge = ab_64 >= 0 ? (1 <<  (max_precision - 1) ) : (1 - (1 << (max_precision - 1)));
    int32_t ab_x2_high32 = static_cast<int32_t>((ab_64 + nudge) / (1ll << max_precision));
    return overflow ? std::numeric_limits < int32_t > ::max() : ab_x2_high32;
}

static int32_t MultiplyByQuantizedMultiplierSmallerThanOne(int32_t x,
         int32_t quantized_multiplier, int right_shift, bool p) {

    return RoundingDivideByPOT(
        SaturatingRoundingDoublingHighMul(x, quantized_multiplier, p),
        right_shift, p);
}
static int32_t MultiplyByQuantizedMultiplierSmallerThanOne_new(int32_t x,
                                                        int32_t quantized_multiplier, int right_shift, bool p) {

    assert(right_shift >= 0);
    assert(right_shift <= max_precision);

    int64_t a_64(x);
    int64_t b_64(quantized_multiplier);
    //the max_precision value is 31
    int32_t ab = static_cast<int32_t>( ( a_64 * b_64 ) >> max_precision );
    if(p)
        printf("c: %d\n",ab);

    const int32_t mask = Dup((1ll << right_shift) - 1);
    const int32_t zero = Dup(0);
    const int32_t one = Dup(1);
    const int32_t remainder = BitAnd(ab, mask);
    const int32_t threshold = ShiftRight(mask, 1);

    //if remainder > threshold res+1 else res+0 ( res = ShiftRight(x, exponent) )
    return Add(ShiftRight(ab, right_shift),
            BitAnd(MaskIfGreaterThan(remainder, threshold), one));

}

static int8_t quantized_src(int32_t in_value, int32_t in_mult, int32_t in_shift, bool p)
{
    if(in_value < 0){
        int8_t dstv = -128;
        return dstv;
    }
    int32_t eight_bit_res = MultiplyByQuantizedMultiplierSmallerThanOne_new(in_value, in_mult, in_shift, p);
    eight_bit_res = std::min(eight_bit_res, action_max_fb);
    int8_t dst = eight_bit_res - 128;
    return dst;
}

#define mul_and_expand(vsum, vin, vfilter, offset)                  \
	do                                                              \
	{                                                               \
	v8i16 vmulsr = __msa_mulsr_h(vin, vfilter);                 \
	v8i16 vmulsl = __msa_mulsl_h(vin, vfilter);                 \
	vsum[0 + offset] = __msa_accsr_w(vsum[0 + offset], vmulsr); \
	vsum[1 + offset] = __msa_accsl_w(vsum[1 + offset], vmulsr); \
	vsum[2 + offset] = __msa_accsr_w(vsum[2 + offset], vmulsl); \
	vsum[3 + offset] = __msa_accsl_w(vsum[3 + offset], vmulsl); \
	} while (0)

static inline void v_quantized(
     v4i32 *v_mul, v4i32 *v_shift, int8_t *out, v4i32 *sum,
     v8i16 action_max, v4i32 action_min, v16i8 quantize_offset) {
    sum[0] = __msa_max_s_w(sum[0], action_min);
    sum[1] = __msa_max_s_w(sum[1], action_min);
    sum[2] = __msa_max_s_w(sum[2], action_min);
    sum[3] = __msa_max_s_w(sum[3], action_min);

    sum[0] = __msa_mul_q_w(sum[0], v_mul[0]);
    sum[1] = __msa_mul_q_w(sum[1], v_mul[1]);
    sum[2] = __msa_mul_q_w(sum[2], v_mul[2]);
    sum[3] = __msa_mul_q_w(sum[3], v_mul[3]);

    sum[0] = __msa_srar_w(sum[0], v_shift[0]);
    sum[1] = __msa_srar_w(sum[1], v_shift[1]);
    sum[2] = __msa_srar_w(sum[2], v_shift[2]);
    sum[3] = __msa_srar_w(sum[3], v_shift[3]);

    // <= 8bit
    v8i16 v_sum16[2];
    v_sum16[0] = __msa_satss_h(sum[1], sum[0]);
    v_sum16[1] = __msa_satss_h(sum[3], sum[2]);

    v_sum16[0] = __msa_min_s_h(v_sum16[0], action_max);
    v_sum16[1] = __msa_min_s_h(v_sum16[1], action_max);

    v16i8 v_sum8 = __msa_pckev_b((v16i8)v_sum16[1],(v16i8)v_sum16[0]);

    v_sum8 -= quantize_offset;
    __msa_st_b(v_sum8, out, 0);
}

static void LSTM(int8_t *input_data, int8_t *filter_data,
			 int32_t *bias_data, int8_t *output_data0,
			int32_t *mul_data_pos, int32_t *shift_data_pos,
			int input_depth, int output_depth,
			int input_height, int input_width,
			int output_height, int output_width,
			int kernel_height, int kernel_width, int* strides)
{
	int8_t *in_data = input_data;
	int8_t *ft_data = filter_data;
	int8_t *output_data = output_data0;
	int32_t *bias = bias_data;
	int32_t *mul_pos = mul_data_pos;
	int32_t *shift_pos = shift_data_pos;

	int8_t *out_ptr;
	int8_t *in_ptr;
	int8_t *in_loc;
	int8_t *in_locr;
	int8_t *ft_ptr;
	int8_t *ft_loc;
	int8_t *ft_locr;
	int in_h;
	int in_w;
	int out_w;
	int out_h;
	int i, j, p, l, fi = 0, fj = 0;
	int eight_bit_res;
	int sum1;
	int filter_count = output_depth;

	for (out_h = 0; out_h < output_height; ++out_h)
		for (out_w = 0; out_w < output_width; out_w++)
		{
			out_ptr = output_data + (out_h * output_width + out_w) * output_depth;

			in_h = out_h * strides[0];
			in_w = out_w * strides[1];

			ft_ptr = ft_data;
			in_ptr = in_data + (in_h * input_width + in_w) * input_depth;
			j = 0;
			for (; j <= output_depth - 64; j += 64)
			{
				in_loc = in_ptr + j;
				ft_loc = ft_ptr + j;
				v4i32 vsum[16];
				vsum[0] = __msa_ld_w(bias + j, 0);
				vsum[1] = __msa_ld_w(bias + j, 16);
				vsum[2] = __msa_ld_w(bias + j, 32);
				vsum[3] = __msa_ld_w(bias + j, 48);
				vsum[4] = __msa_ld_w(bias + j, 64);
				vsum[5] = __msa_ld_w(bias + j, 80);
				vsum[6] = __msa_ld_w(bias + j, 96);
				vsum[7] = __msa_ld_w(bias + j, 112);
				vsum[8] = __msa_ld_w(bias + j, 128);
				vsum[9] = __msa_ld_w(bias + j, 144);
				vsum[10] = __msa_ld_w(bias + j, 160);
				vsum[11] = __msa_ld_w(bias + j, 176);
				vsum[12] = __msa_ld_w(bias + j, 192);
				vsum[13] = __msa_ld_w(bias + j, 208);
				vsum[14] = __msa_ld_w(bias + j, 224);
				vsum[15] = __msa_ld_w(bias + j, 240);
				for (fi = 0; fi < kernel_height; ++fi)
				{
				 in_locr = in_loc + fi * (input_width * input_depth);
					ft_locr = ft_loc + fi * (kernel_height * filter_count);
		for (fj = 0; fj < kernel_width; ++fj)
		{
		v16i8 vin[4], vft[4];
		vin[0] = (v4i32)__msa_ld_b(in_locr + fj * input_depth, 0);
		vin[1] = (v4i32)__msa_ld_b(in_locr + fj * input_depth, 16);
		vin[2] = (v4i32)__msa_ld_b(in_locr + fj * input_depth, 32);
		vin[3] = (v4i32)__msa_ld_b(in_locr + fj * input_depth, 48);
		vft[0] = (v4i32)__msa_ld_b(ft_locr + fj * filter_count, 0);
		vft[1] = (v4i32)__msa_ld_b(ft_locr + fj * filter_count, 16);
		vft[2] = (v4i32)__msa_ld_b(ft_locr + fj * filter_count, 32);
	       vft[3] = (v4i32)__msa_ld_b(ft_locr + fj * filter_count, 48);
		mul_and_expand(vsum, vin[0], vft[0], 0);
		mul_and_expand(vsum, vin[1], vft[1], 4);
		mul_and_expand(vsum, vin[2], vft[2], 8);
		mul_and_expand(vsum, vin[3], vft[3], 12);
					}
				}
                {
                    v4i32 v_mul[16];
                    int32_t *p_mul_pos = mul_pos + j;
                    v_mul[0] = __msa_ld_w(p_mul_pos, 0);
                    v_mul[1] = __msa_ld_w(p_mul_pos, 16);
                    v_mul[2] = __msa_ld_w(p_mul_pos, 32);
                    v_mul[3] = __msa_ld_w(p_mul_pos, 48);
                    v_mul[4] = __msa_ld_w(p_mul_pos, 64);
                    v_mul[5] = __msa_ld_w(p_mul_pos, 80);
                    v_mul[6] = __msa_ld_w(p_mul_pos, 96);
                    v_mul[7] = __msa_ld_w(p_mul_pos, 112);
                    v_mul[8] = __msa_ld_w(p_mul_pos, 128);
                    v_mul[9] = __msa_ld_w(p_mul_pos, 144);
                    v_mul[10] = __msa_ld_w(p_mul_pos, 160);
                    v_mul[11] = __msa_ld_w(p_mul_pos, 176);
                    v_mul[12] = __msa_ld_w(p_mul_pos, 192);
                    v_mul[13] = __msa_ld_w(p_mul_pos, 208);
                    v_mul[14] = __msa_ld_w(p_mul_pos, 224);
                    v_mul[15] = __msa_ld_w(p_mul_pos, 240);

                    v4i32 v_shift[16];
                    int32_t *p_shift_pos = shift_pos + j;
                    v_shift[0] = __msa_ld_w(p_shift_pos, 0);
                    v_shift[1] = __msa_ld_w(p_shift_pos, 16);
                    v_shift[2] = __msa_ld_w(p_shift_pos, 32);
                    v_shift[3] = __msa_ld_w(p_shift_pos, 48);
                    v_shift[4] = __msa_ld_w(p_shift_pos, 64);
                    v_shift[5] = __msa_ld_w(p_shift_pos, 80);
                    v_shift[6] = __msa_ld_w(p_shift_pos, 96);
                    v_shift[7] = __msa_ld_w(p_shift_pos, 112);
                    v_shift[8] = __msa_ld_w(p_shift_pos, 128);
                    v_shift[9] = __msa_ld_w(p_shift_pos, 144);
                    v_shift[10] = __msa_ld_w(p_shift_pos, 160);
                    v_shift[11] = __msa_ld_w(p_shift_pos, 176);
                    v_shift[12] = __msa_ld_w(p_shift_pos, 192);
                    v_shift[13] = __msa_ld_w(p_shift_pos, 208);
                    v_shift[14] = __msa_ld_w(p_shift_pos, 224);
                    v_shift[15] = __msa_ld_w(p_shift_pos, 240);
                    v8i16 action_max = __msa_fill_h(255);
                    v4i32 action_min = __msa_fill_w(0);
                    v16i8 quantize_offset = __msa_fill_b(128);
                    v_quantized(&v_mul[0], &v_shift[0], out_ptr+j, &vsum[0], action_max, action_min, quantize_offset);
                    v_quantized(&v_mul[4], &v_shift[4], out_ptr+j+16, &vsum[4], action_max, action_min, quantize_offset);
                    v_quantized(&v_mul[8], &v_shift[8], out_ptr+j+32, &vsum[8], action_max, action_min, quantize_offset);
                    v_quantized(&v_mul[12], &v_shift[12], out_ptr+j+48, &vsum[12], action_max, action_min, quantize_offset);
                }
			}
			for (; j <= output_depth - 32; j += 32)
			{
				in_loc = in_ptr + j;
				ft_loc = ft_ptr + j;
				v4i32 vsum[8];
				vsum[0] = __msa_ld_w(bias + j, 0);
				vsum[1] = __msa_ld_w(bias + j, 16);
				vsum[2] = __msa_ld_w(bias + j, 32);
				vsum[3] = __msa_ld_w(bias + j, 48);
				vsum[4] = __msa_ld_w(bias + j, 64);
				vsum[5] = __msa_ld_w(bias + j, 80);
				vsum[6] = __msa_ld_w(bias + j, 96);
				vsum[7] = __msa_ld_w(bias + j, 112);
				for (fi = 0; fi < kernel_height; ++fi)
				{
			in_locr = in_loc + fi * (input_width * input_depth);
			ft_locr = ft_loc + fi * (kernel_width * filter_count);
		for (fj = 0; fj < kernel_width; ++fj)
	{
		v16i8 vin[2], vft[2];
		vin[0] = (v4i32)__msa_ld_b(in_locr + fj * input_depth, 0);
		vin[1] = (v4i32)__msa_ld_b(in_locr + fj * input_depth, 16);
		vft[0] = (v4i32)__msa_ld_b(ft_locr + fj * filter_count, 0);
		vft[1] = (v4i32)__msa_ld_b(ft_locr + fj * filter_count, 16);
		mul_and_expand(vsum, vin[0], vft[0], 0);
		mul_and_expand(vsum, vin[1], vft[1], 4);
					}
				}
                {
                    v4i32 v_mul[8];
                    int32_t *p_mul_pos = mul_pos + j;
                    v_mul[0] = __msa_ld_w(p_mul_pos, 0);
                    v_mul[1] = __msa_ld_w(p_mul_pos, 16);
                    v_mul[2] = __msa_ld_w(p_mul_pos, 32);
                    v_mul[3] = __msa_ld_w(p_mul_pos, 48);
                    v_mul[4] = __msa_ld_w(p_mul_pos, 64);
                    v_mul[5] = __msa_ld_w(p_mul_pos, 80);
                    v_mul[6] = __msa_ld_w(p_mul_pos, 96);
                    v_mul[7] = __msa_ld_w(p_mul_pos, 112);

                    v4i32 v_shift[8];
                    int32_t *p_shift_pos = shift_pos + j;
                    v_shift[0] = __msa_ld_w(p_shift_pos, 0);
                    v_shift[1] = __msa_ld_w(p_shift_pos, 16);
                    v_shift[2] = __msa_ld_w(p_shift_pos, 32);
                    v_shift[3] = __msa_ld_w(p_shift_pos, 48);
                    v_shift[4] = __msa_ld_w(p_shift_pos, 64);
                    v_shift[5] = __msa_ld_w(p_shift_pos, 80);
                    v_shift[6] = __msa_ld_w(p_shift_pos, 96);
                    v_shift[7] = __msa_ld_w(p_shift_pos, 112);
                    v8i16 action_max = __msa_fill_h(255);
                    v4i32 action_min = __msa_fill_w(0);
                    v16i8 quantize_offset = __msa_fill_b(128);
                    v_quantized(&v_mul[0], &v_shift[0], out_ptr+j, &vsum[0], action_max, action_min, quantize_offset);
                    v_quantized(&v_mul[4], &v_shift[4], out_ptr+j+16, &vsum[4], action_max, action_min, quantize_offset);
                }
			}
			for (; j <= output_depth - 16; j += 16)
			{
				in_loc = in_ptr + j;
				ft_loc = ft_ptr + j;
				v4i32 vsum[4];
				vsum[0] = __msa_ld_w(bias + j, 0);
				vsum[1] = __msa_ld_w(bias + j, 16);
				vsum[2] = __msa_ld_w(bias + j, 32);
				vsum[3] = __msa_ld_w(bias + j, 48);
				for (fi = 0; fi < kernel_height; ++fi)
				{
			in_locr = in_loc + fi * (input_width * input_depth);
			ft_locr = ft_loc + fi * (kernel_width * filter_count);
			for (fj = 0; fj < kernel_width; ++fj)
			{
			v16i8 vin, vft;
			vin = (v4i32)__msa_ld_b(in_locr + fj * input_depth, 0);
			vft = (v4i32)__msa_ld_b(ft_locr + fj * filter_count, 0);
			mul_and_expand(vsum, vin, vft, 0);
			}
		}
                {
                    v4i32 v_mul[4];
                    int32_t *p_mul_pos = mul_pos + j;
                    v_mul[0] = __msa_ld_w(p_mul_pos, 0);
                    v_mul[1] = __msa_ld_w(p_mul_pos, 16);
                    v_mul[2] = __msa_ld_w(p_mul_pos, 32);
                    v_mul[3] = __msa_ld_w(p_mul_pos, 48);

                    v4i32 v_shift[4];
                    int32_t *p_shift_pos = shift_pos + j;
                    v_shift[0] = __msa_ld_w(p_shift_pos, 0);
                    v_shift[1] = __msa_ld_w(p_shift_pos, 16);
                    v_shift[2] = __msa_ld_w(p_shift_pos, 32);
                    v_shift[3] = __msa_ld_w(p_shift_pos, 48);

                    v8i16 action_max = __msa_fill_h(255);
                    v4i32 action_min = __msa_fill_w(0);
                    v16i8 quantize_offset = __msa_fill_b(128);
                    v_quantized(v_mul, v_shift, out_ptr+j, &vsum[0], action_max, action_min, quantize_offset);
                }
			}
			for (; j < output_depth; ++j)
			{
				in_loc = in_ptr + j;
				ft_loc = ft_ptr + j;
				sum1 = 0;
				for (fi = 0; fi < kernel_height; ++fi)
				{
			int inh = in_h + fi;
			in_locr = in_loc + fi * (input_width * input_depth);
			ft_locr = ft_loc + fi * (kernel_width * filter_count);
			for (fj = 0; fj < kernel_width; ++fj)
			{
				int inw = in_w + fj;
				if ((inh < input_height) && (inw < input_width))
			sum1 += ft_locr[fj * filter_count] * in_locr[fj * input_depth];
			}
				}
				sum1 += bias[j];
                out_ptr[j] = quantized_src(sum1, mul_pos[j], shift_pos[j], false);
			}
		}
}

Register B(1, forward_msa);
附录五 共享内存数据结构和函数调用接口
struct TransData {
  void *data;
  int width;
  int height;
  int stride;
  int flags;
  TransData(){
    flags = -2;
  }
  TransData(void *_d,int w,int h,int stride_,int flags_){
    data = _d;
    stride = stride_;
    width = w;
    height = h;
    flags = flags_;
  }
  TransData(void *_d,int w,int h,int flags_){
    data = _d;
    width = w;
    height = h;
    flags = flags_;
  }
  TransData(int w,int h){
    width = w;
    height = h;
  }
  TransData(int w,int h,int stride_){
    stride = stride_;
    width = w;
    height = h;
  }
  void Dump(){
    printf("----------------------------\n");
    printf("data:   %p\n",data);
    printf("width:  %d\n",width);
    printf("height: %d\n",height);
    printf("stride: %d\n",stride);
    printf("flags:  %d\n",flags);
    printf("----------------------------\n");
  }
};

class RetrievalData{
 public:
  // 输入
  virtual bool push_back(){
    return false;
  };
  virtual void *back(TransData* param = NULL){
    return NULL;
  };

  // 输出
  virtual void* front(TransData* param = NULL){
    return NULL;
  };
  virtual bool pop_front(){
    return false;
  };
  //参数
  virtual TransData getParam(){
    return TransData();
  }
  virtual bool setParam(TransData tran)
  {
    return false;
  }
  // 控制
  virtual bool stop() = 0;
  virtual void reset() = 0;

};
extern "C" {
  std::shared_ptr<RetrievalData> OcrJzInstance();
  std::shared_ptr<RetrievalData> OcrLiteInstance();
  std::shared_ptr<RetrievalData> StitcherInstance(int width,int height);
  std::shared_ptr<RetrievalData> StitcherInstance2(int width,int height, int rleft,int rtop,int rright,int rbottom,int enableAutoLum);
  std::shared_ptr<RetrievalData> WordLineInstance(int width,int height);
};」

 template<typename T>
class DataTransfer {
 private:
  std::mutex mtx;

  volatile bool ready;
  std::condition_variable cvReady;
  volatile bool mStop;
  volatile int state;
  int dataCount;
  const string mName;
  function<void(T&)> callbackFunc;

 public:
  DataTransfer(int count,string name):mName(name){
    // std::cout << mName << " push start" << std::endl;
    mStop = false;
    dataCount = count;
    ready = false;
    state = 0;
  }

  DataTransfer(function<void(T&)> callback_,string name):callbackFunc(callback_),mName(name){
    mStop = false;
    callbackFunc = callback_;
    dataCount = -1;
    state = 0;
  }
  void push(T& d){
    std::unique_lock <std::mutex> lck(mtx);
    if(dataCount == -1){
      callbackFunc(d);
    }else{
      if(ready == false){
        ready = true;
        cvReady.notify_one();
      }
    }
  }
  bool front(T& d){
    std::unique_lock <std::mutex> lck(mtx);
    while(ready == false && state == 0)
      cvReady.wait(lck);
    //std::cout << mName << " front " << endl;
    ready = false;
    return !mStop;
  }
  void pop(T& d){
    state &= ~1;
  }
  void finish()
  {
    std::unique_lock <std::mutex> lck(mtx);
    state |= 1;
    cvReady.notify_one();
  }
  bool isFinish()
  {
    std::unique_lock <std::mutex> lck(mtx);
    return (state&1) == 1;
  }
  void reset()
  {
    std::unique_lock <std::mutex> lck(mtx);
    // std::cout << mName << " reset " << dataList.size() << std::endl;
    ready = false;
    mStop = false;
    // dataList.clear();
  }
  void stop(){
    std::unique_lock <std::mutex> lck(mtx);
    // std::cout << mName << " stop " << dataList.size() << std::endl;
    mStop = true;

  }
  bool isEmpty()
  {
    std::unique_lock <std::mutex> lck(mtx);
    return ready == false;
  }
};


