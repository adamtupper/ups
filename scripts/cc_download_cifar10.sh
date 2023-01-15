#!/bin/bash
export VAR_MD5=c58f30108f718f92721af3b95e74349a

mkdir -p $project/data

/usr/bin/wget -O $project/data/cifar-10-python.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
echo "$VAR_MD5  ./cifar-10-python.tar.gz" | md5sum -c -
