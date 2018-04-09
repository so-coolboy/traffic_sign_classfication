# traffic_sign_classfication
使用cNN对交通标志进行分类
参考了一位大神的博客  ：http://www.cnblogs.com/skyfsm/p/8051705.html

训练集使用：
python train.py --data_train ./train --data_test ./test --model traffic_sign.model

单张图片测试：
python predict.py --model traffic_sign.model -i ./test/00000/00017_00000.png.png -s

