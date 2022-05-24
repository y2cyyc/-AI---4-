import paddle


class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet,self).__init__()
        self.resnet = paddle.vision.models.resnet152(pretrained=True, num_classes=0)
        self.flatten = paddle.nn.Flatten()
        self.linear = paddle.nn.Linear(2048, 512)
        self.linear2 = paddle.nn.Linear(512, 8)

    def forward(self, img):
        y = self.resnet(img)
        print(y.shape)
        y = self.flatten(y)
        print(y.shape)
        y = self.linear(y)
        out = self.linear2(y)

        return out


if __name__ == '__main__':
    model = MyNet()

    input_tensor = paddle.randn((4, 3, 512, 512))
    out_put1 = model(input_tensor)
    print(out_put1.shape)
