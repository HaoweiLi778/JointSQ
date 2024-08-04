#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# 将一个文件夹下图片按比例分在三个文件夹下
import os
import random
import shutil
from shutil import copy2
datadir_normal = "/home/wangzixuan/data/NWPU/NWPU-RESISC45/"

def main():
    type_list = os.listdir(datadir_normal)#（图片文件夹）
    for type in type_list:
        data_root = datadir_normal + type + '/'
        all_data = os.listdir(data_root)
        num_all_data = len(all_data)
        print( "num_all_data: " + str(num_all_data) )
        index_list = list(range(num_all_data))
        random.shuffle(index_list)
        num = 0

        # 创建目录
        trainDir = "/home/wangzixuan/data/NWPU/train/" + type +'/'#（将训练集放在这个文件夹下）
        if not os.path.exists(trainDir):
            os.mkdir(trainDir)
                
        validDir = "/home/wangzixuan/data/NWPU/val/" + type +'/'#（将验证集放在这个文件夹下）
        if not os.path.exists(validDir):
            os.mkdir(validDir)

        testDir = "/home/wangzixuan/data/NWPU/test/" + type + '/'  # （将测试集放在这个文件夹下）
        if not os.path.exists(testDir):
            os.mkdir(testDir)
                
        for i in index_list:
            fileName = os.path.join(datadir_normal, type, all_data[i])
            if num < num_all_data*0.8:
                #print(str(fileName))
                copy2(fileName, trainDir)
            elif num>=num_all_data*0.8 and num < num_all_data*0.9:
                #print(str(fileName))
                copy2(fileName, validDir)
            elif num>=num_all_data*0.9 and num < num_all_data:
                # print(str(fileName))
                copy2(fileName, testDir)
            num += 1

if __name__ == '__main__':
    main()
