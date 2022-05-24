import paddle

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import albumentations as A

def train_augmentation():
    train_transform = [
        # A.RandomSizedCrop(min_max_height=(1024, 1024), height=1024, width=1024, w2h_ratio=1.0, always_apply=False, p=1.),
        A.Flip(p=0.5),
        A.Rotate(p=0.5),
        ############################ add #################
        # A.RandomGridShuffle(grid=(2, 2), p=0.2),
        #
        # A.Transpose(p=0.5),
        # A.ShiftScaleRotate(p=0.5),
        ############################ add #################

        # A.OneOf([
        #     A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.),
        #     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=1.),
        #     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.),
        # ], p=0.5),

    ]
    return A.Compose(train_transform)


class MyDateset(paddle.io.Dataset):
    def __init__(self, mode='train', train_imgs_dir='D:/yyc/competition/AIstudio/Document_detection/train_datasets_document_detection_0411/images/', transform=train_augmentation(),
                 label_imgs_dir='D:/yyc/competition/AIstudio/Document_detection/train_datasets_document_detection_0411/segments/', train_txt='D:/yyc/competition/AIstudio/Document_detection/train_datasets_document_detection_0411/data_info.txt'):
        super(MyDateset, self).__init__()

        self.mode = mode
        self.train_imgs_dir = train_imgs_dir
        self.label_imgs_dir = label_imgs_dir

        self.transform = transform

        with open(train_txt, 'r') as f:
            self.train_infor = f.readlines()


    def __getitem__(self, index):
        item = self.train_infor[index][:-1]
        splited = item.split(',')
        img_name = splited[0]

        img = cv2.imread(self.train_imgs_dir + img_name + '.jpg')
        h, w, c = img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        lab = cv2.imread(self.label_imgs_dir + img_name + '.png')

        trans = self.transform(image=img, mask=lab)
        img, lab = trans['image'], trans['mask']


        lab = paddle.vision.transforms.resize(lab, (768, 768), interpolation='nearest')
        lab = (lab[:, :, 0] == 255)

        # plt.imshow(lab)
        # plt.show()


        # 对图片进行resize，调整明暗对比度等参数
        img = paddle.vision.transforms.resize(img, (768, 768), interpolation='bilinear')
        if np.random.rand() < 1 / 2:
            img = paddle.vision.transforms.adjust_brightness(img, np.random.rand() * 2)
        else:
            if np.random.rand() < 1 / 2:
                img = paddle.vision.transforms.adjust_contrast(img, np.random.rand() * 2)
            else:
                img = paddle.vision.transforms.adjust_hue(img, np.random.rand() - 0.5)



        img = img.transpose((2, 0, 1))
        img = img / 255



        img = paddle.to_tensor(img).astype('float32')
        label = paddle.to_tensor(lab).astype('int64')

        return img, label


    def __len__(self):
        return len(self.train_infor)



# 对dataloader进行测试

def main():
    train_dataset=MyDateset()

    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=False)

    for step, data in enumerate(train_dataloader):
        img, label = data
        print(step, img.shape, label.shape)
        break

if __name__ == '__main__':
    main()