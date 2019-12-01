# win10 安装vs2015 + cuda10.1 + cudnn7 顺序安装，vs要先装
vs2015 下载：迅雷
ed2k://|file|en_visual_studio_community_2015_x86_dvd_6847364.iso|3965825024|6A7D8489BB2877E6BB8ACB2DD187B637|/

cuda10.1 特定版本下载url：
https://developer.nvidia.com/cuda-toolkit-archive

cudnn7 下载需登录，qq登录
https://developer.nvidia.com/rdp/cudnn-archive

# pytorch下载 官网查找下载，可用迅雷更快
https://anaconda.org/pytorch/pytorch/1.3.1/download/win-64/pytorch-1.3.1-py3.7_cuda101_cudnn7_0.tar.bz2
本地conda安装命令：
conda install --use-local pytorch-1.3.1-py3.7_cuda101_cudnn7_0.tar.bz2

# 其余安装包下载
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

#ref
https://blog.csdn.net/chenxaioxue/article/details/81634536
