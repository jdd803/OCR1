# OCR1
本项目中训练代码为train1.py,由于仅使用40张图片作为训练集，所以并没有验证模型的泛化能力。
项目使用TensorFlow2.0.0-alpha0实现，需要NVIDIA显卡驱动版本在410.45以上
软件环境： cudnn7.4.2
          cuda 10.130
          
数据集：dataset
      其中gt表示groundtruth box，表示为（x1,y1,x2,y2,1),为左上角坐标和右下角坐标和文本类别1
      mask表示图像中的所有文本mask位置，表示为（x1,y1,x2,y2,x3,y3,x4,y4,c），分别为左上角，右上角，右下角，左下角坐标，以及文本内容（未使用）
      
model1.py:模型定义文件
config:RPN 阶段使用的参数定义文件（来自faster rcnn)
Resnet-50 特征提取模块：Resnet/resnet50.py
lodedata.py:读取文件和对应的gt,mask
lib/Inception/Inception_text1.py:Inception_text模块
lib/ROI_proposal/:包含RPN阶段所需要的所有张量操作
lib/RPN/:包含RPN模块，产生所有anchor以及anchor的偏移量
lib/bbox/:对box进行的处理操作，包括不同格式坐标变换，预测值与真实值之间的偏差等等。
lib/deform_conv_layer/:可变形卷积层，产生经过偏移后的特征图
lib/deform_psroi_pooling/:deformable psroi pooling层
lib/loss/loss_function1.py:模型的5个损失函数定义
lib/nms/nms_v21:查找所有与未被抑制的box重合度超过0.5的被抑制的box

