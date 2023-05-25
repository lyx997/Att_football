# Google Research Football Offensive and Deffensive Attention Agent

## 监督模型训练
```bash
python3 sup_train.py
```
- 在sup_encoders文件夹里自定义不同的观测特征
- 在sup_models文件夹里自定义不同的编码器网络结构

## 强化学习训练
```bash
python3 train_conv1_att.py
```
- 在sup_rewarders文件夹里选择不同的总奖励函数进行训练
- 在sup_encoders文件夹里自定义不同的观测特征
- 在sup_models文件夹里自定义不同的编码器网络结构
- 在algos文件夹里自定义不同的强化学习训练算法

## 模型测试
```bash
python3 new_review.py
```



