import os
import sys
import glob
import json
import cv2
import paddle
from model.swim_b import swimb_fpn
import numpy as np


def process(src_image_dir, save_dir):

    model = swimb_fpn()
    model_state_dict = paddle.load('./model/model_137.pdparams')
    model.set_state_dict(model_state_dict)
    model.eval()

    # json_results = []

    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for image_path in image_paths:
        # do something
        img = cv2.imread(image_path)
        h, w, c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = paddle.vision.transforms.resize(img, (512, 512), interpolation='bilinear')
        img = img.transpose((2, 0, 1))
        img = img / 255
        img = paddle.to_tensor(img).astype('float32')
        img = img.unsqueeze(0)
        with paddle.no_grad():
            s_predict_seg = model(img)

            s_image_h = paddle.flip(img, axis=[3])
            s_predict_h = model(s_image_h)
            s_predict_h_seg = paddle.flip(s_predict_h, axis=[3])

            s_image_v = paddle.flip(img, axis=[2])
            s_predict_v = model(s_image_v)
            s_predict_v_seg = paddle.flip(s_predict_v, axis=[2])

            pre = 0.5 * s_predict_seg + 0.25 * s_predict_h_seg + 0.25 * s_predict_v_seg

        # 保存结果图片
        out_image = paddle.argmax(pre, axis=1).astype(float)
        # print(out_image[0,200:300,200:300])
        out_image[out_image == 1] = 255
        out_image = paddle.fluid.layers.expand(out_image, expand_times=[3, 1, 1])
        # print(out_image[0,200:300,200:300])
        out_image = out_image.transpose((1, 2, 0)).numpy().astype(np.uint8)

        out_image = paddle.vision.transforms.resize(out_image, (h, w), interpolation='bilinear')


        save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        cv2.imwrite(save_path, out_image)

    # 或者保存坐标信息到json文件



if __name__ == "__main__":
    # assert len(sys.argv) == 3
    #
    # src_image_dir = sys.argv[1]
    # save_dir = sys.argv[2]

    # src_image_dir = r'D:\yyc\competition\AIstudio\Document_detection\testA_datasets_document_detection\images'
    src_image_dir = r'D:\yyc\competition\AIstudio\Document_detection\document_detection_testB_dataset'
    save_dir = r'D:\yyc\competition\AIstudio\Document_detection\results'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    process(src_image_dir, save_dir)