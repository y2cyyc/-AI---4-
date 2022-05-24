import paddle
from model.get_model import get_swim, get_swimb, get_swiml
from dataloader3 import MyDateset
import os
import time
from paddle.nn.functional import dice_loss, softmax_with_cross_entropy
import matplotlib.pyplot as plt
import numpy as np
from paddle import nn
from losses.lovasz_loss import LovaszSoftmaxLoss
from losses.dice_loss import DiceLoss
from losses.ohem_cross_entropy_loss import OhemCrossEntropyLoss
from losses.semantic_connectivity_loss import SemanticConnectivityLoss
from losses.detail_aggregate_loss import DetailAggregateLoss

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
paddle.set_device('gpu')
batch_s = 4
output_model_dir = './swimb'
if not os.path.exists(output_model_dir):
    os.mkdir(output_model_dir)

# log file
file_path = './swimb/train_log.txt'
f_log = open(file_path, 'a')

######## model
# model = get_unetrw()
# model = get_swiml()
model = get_swimb()
model.train()

train_dataset=MyDateset()

# 需要接续之前的模型重复训练可以取消注释
param_dict = paddle.load('./swimb/model_125.pdparams')
model.load_dict(param_dict)
# model_pretrain_dict = paddle.load('./models_para/model_19.pdparams')
# model_dict = model.state_dict()
# pretrained_dict = {k: v for k, v in model_pretrain_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.load_dict(model_dict)

train_dataloader = paddle.io.DataLoader(
    train_dataset,
    batch_size=batch_s,
    shuffle=True,
    drop_last=False)


max_epoch=150
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.0016, T_max=max_epoch)
opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())

loss_lvz = LovaszSoftmaxLoss()
loss_dice = DiceLoss()
loss_ohem = OhemCrossEntropyLoss()
loss_sc = SemanticConnectivityLoss()
loss_dal = DetailAggregateLoss()

start_epoch = 126
epoch_steps = train_dataset.__len__()//batch_s
print('epoch_steps', epoch_steps)
now_step = start_epoch*epoch_steps

for epoch in range(start_epoch, max_epoch):
    f_log.write('epoch:'+str(epoch)+'\n')
    for step, data in enumerate(train_dataloader):
        now_step+=1

        img, label = data

        # label_sig = (img == label).sum(1).astype(float)
        # label_sig[label_sig != 3] = 1
        # label_sig[label_sig == 3] = 0
        # label_sig = label_sig.astype(paddle.int64)
        # plt.imshow(paddle.transpose(img[0, :, :, :], [1, 2, 0]).numpy())
        # plt.show()
        # plt.imshow(paddle.transpose(label[0,:,:,:], [1,2,0]).numpy())
        # plt.show()
        #
        # plt.imshow(np.abs(paddle.transpose(img[0, :, :, :] -label[0,:,:,:] , [1, 2, 0]).numpy()))
        # plt.show()
        #
        # plt.imshow(label_sig[0,:,:].numpy())
        # plt.show()

        pre = model(img)
        # print(np.unique(paddle.argmax(nn.Softmax(axis=1)(pre), axis=1)[0,:,:].numpy() * 255))
        #
        # plt.imshow(paddle.argmax(nn.Softmax(axis=1)(pre), axis=1)[0,:,:].numpy() * 255)
        # plt.show()

        # loss = softmax_with_cross_entropy(paddle.transpose(pre, [0, 2, 3, 1]), label_sig.unsqueeze(3))
        loss1 = loss_lvz(pre, label)
        loss2 = loss_dice(pre, label)
        # loss3 = loss_ohem(pre, label)
        # loss4 = loss_dal(pre, label)
        # loss = (loss1 + loss2) * 0.5 #+ loss4
        loss = 0.65 * loss1 + loss2 * 0.35

        # loss1 = lossfn(pre,label).mean()
        # loss2 = losspsnr(pre,label).mean()
        # loss = (loss1+loss2/100)/2
        # loss = loss2

        loss.backward()
        opt.step()
        opt.clear_gradients()

        scheduler.step(epoch + step/epoch_steps)

        lr_update = opt.get_lr()
        if now_step%10==0:
            content_log = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())) + ' ' + "epoch: {}, batch: {}, loss is: {}, lossohem is: {},lr:{}".format(epoch, step, ((loss1 + loss2) * 0.5).mean().numpy(), loss.mean().numpy(),lr_update)
            print(content_log)
            f_log.write(content_log + '\n')
    paddle.save(model.state_dict(), os.path.join(output_model_dir, 'model_' + str(epoch)+'.pdparams'))
f_log.close()
paddle.save(model.state_dict(), 'model.pdparams')

