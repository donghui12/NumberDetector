import torch
import cv2 as cv
import numpy as np
from model import Net

drawing = False  # 是否开始画图
mode = False  # True：画矩形，False：画圆
start = (-1, -1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_image(img, net):
    resized_img = cv.resize(img, (28, 28))
    gray_img = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    image = torch.tensor([gray_img, ], dtype=torch.uint8)
    image = image.unsqueeze(1)
    image = image.float()
    print(image.size())
    image.to(device)
    output = net(image)
    pred = output.argmax(dim=1, keepdim=True)
    print(pred)

# 鼠标的回调函数的参数格式是固定的，不要随意更改。
def mouse_event(event, x, y, flags, param):
    global start, drawing, mode

    # 左键按下：开始画图
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start = (x, y)
    # 鼠标移动，画图
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            if mode:
                cv.rectangle(img, start, (x, y), (0, 255, 0), -1)
            else:
                cv.circle(img, (x, y), 5, (0, 0, 255), -1)
    # 左键释放：结束画图
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        if mode:
            cv.rectangle(img, start, (x, y), (0, 255, 0), -1)
        else:
            cv.circle(img, (x, y), 5, (0, 0, 255), -1)


img = np.zeros((256, 256, 3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image', mouse_event)
root = './NumberDetecor/datasets'

net = Net()
net.load_state_dict(torch.load(root+'./mnist_cnn.pt'))

while(True):
    cv.imshow('image', img)
    # 按下m切换模式
    if cv.waitKey(1) == ord('m'):
        mode = not mode
    elif cv.waitKey(1) == ord('y'):
        process_image(img, net)
        img = np.zeros((256, 256, 3), np.uint8)
        
    # 按ESC键退出程序
    elif cv.waitKey(1) == 27:
        break