//
// Created by 田一菲 on 2019/1/12.
//
#include<opencv2/opencv.hpp>
#include<iostream>
#include "MSDB.h"
#include "CLAHE_DWT.h"

using namespace std;
using namespace cv;



void hist(const Mat &src,int i){
    Mat imageRGB[3];
    split(src, imageRGB);
    for (int i = 0; i < 3; i++)
    {
        equalizeHist(imageRGB[i], imageRGB[i]);
    }
    Mat dst;
    merge(imageRGB, 3, dst);
    //imshow("直方图均衡化图像增强效果", dst);
    //waitKey();
    imwrite("output/Hist/"+ to_string(i) + ".png", dst);      //写出图像
}

void laplace(const Mat& src, int i){
    Mat dst;
    Mat kernel = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    filter2D(src, dst, CV_8UC3, kernel);
    imwrite("output/Laplace/"+ to_string(i) + ".png", dst);      //写出图像
}


void gamma(const Mat& src,int i){
    Mat dst(src.size(), CV_32FC3);
    double r = 0.5;
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            dst.at<Vec3f>(i, j)[0] = (float)pow((src.at<Vec3b>(i, j)[0]),r);
            dst.at<Vec3f>(i, j)[1] = (float)pow((src.at<Vec3b>(i, j)[1]),r);
            dst.at<Vec3f>(i, j)[2] = (float)pow((src.at<Vec3b>(i, j)[2]),r);
        }
    }
    //归一化到0~255
    normalize(dst, dst, 0, 255, CV_MINMAX);
    //转换成8bit图像显示
    convertScaleAbs(dst, dst);
    imwrite("output/Gamma/"+ to_string(i) + ".png", dst);      //写出图像
}



int main()
{
    Mat src, dst1, dst2;
    namedWindow("src",0);
    namedWindow("dst",0);
    for(int i= 1;i<=5;i++){
        src = imread("input/"+ to_string(i) + ".png", IMREAD_COLOR);  //以彩色的方式读入图像

        if (!src.data)               //判断图像是否被正确读取；
        {
            return 0;
        }

        hist(src,i);
        laplace(src,i);
        gamma(src,i);

        dst1 = MultiScaleDetailBoosting(src,5);
        imwrite("output/MSDB/"+ to_string(i) + ".png", dst1);      //写出图像

        dst2 = CLAHE_DWT(src);
        imwrite("output/CLAHE_DWT/"+ to_string(i) + ".png", dst2);      //写出图像

        cout<<i<<endl;
    }
}


