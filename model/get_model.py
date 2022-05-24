from model.res152 import MyNet
from model.swim_unet import swim_fpn
from model.swim_b import swimb_fpn
from model.swim_l import swiml_fpn
from model.swim_b_linear import swimb_fpn_linear

def get_res152():
    return MyNet()

def get_swim():
    return swim_fpn()


def get_swimb():
    return swimb_fpn(pretrain=r'D:\yyc\yycpython\Document_seg\model\SwinTransformer_base_patch4_window12_384_22kto1k_pretrained.pdparams')


def get_swiml():
    return swiml_fpn(pretrain=r'D:\yyc\yycpython\Document_seg\model\SwinTransformer_large_patch4_window12_384_22kto1k_pretrained.pdparams')


def get_swimb_linear():
    return swimb_fpn_linear(pretrain1=r'D:\yyc\yycpython\Document_seg\model/model_29.pdparams', pretrain2=r'D:\yyc\yycpython\Document_seg\model\SwinTransformer_base_patch4_window12_384_22kto1k_pretrained.pdparams')

