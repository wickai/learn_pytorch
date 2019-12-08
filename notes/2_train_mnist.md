# mnist数据集下载
## 自动下载太慢
> http://yann.lecun.com/exdb/mnist/
>
> 修改文件 C:\Users\wickai\Anaconda3\envs\py3_torch\Lib\site-packages\torchvision\datasets\mnist.py
> urls变量
>
```python
# urls = [
#     'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
#     'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
#     'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
#     'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
# ]
urls = [
    'file:///E:/downloads/datasets/mnist/train-images-idx3-ubyte.gz',
    'file:///E:/downloads/datasets/mnist/train-labels-idx1-ubyte.gz',
    'file:///E:/downloads/datasets/mnist/t10k-images-idx3-ubyte.gz',
    'file:///E:/downloads/datasets/mnist/t10k-labels-idx1-ubyte.gz',
]
```

# 训练
```shell script
python image_classify/mnist.py --epochs=1
```
参数详解： （仅适用该脚本训练）
- --batch-size： 小批次梯度下降训练的批次大小
- --test-batch-size： 测试的批次大小
- --epochs： 迭代轮数
- --gamma: 学习率衰减gamma值
- --no-cuda: 是否不用cuda，默认false
- --seed: 随机初始化种子，用于控制相同初始化，默认1
- --log-interval: 多少minibatch打log
- --save-model: 是否保存模型，默认false

# 更多demo
https://github.com/pytorch/examples

