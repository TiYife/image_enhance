//
// Created by 田一菲 on 2019/1/12.
//

#ifndef IMAGE_ENHANCE_CLAHE_DWT_H
#define IMAGE_ENHANCE_CLAHE_DWT_H
# include<opencv2/opencv.hpp>
# include<iostream>

using namespace std;
using namespace cv;
int depth = 1;
double alpha = 0.1,beta = 1.5;

Mat get_gray(Mat src, int channel){
    int cols = src.cols;
    int rows = src.rows;
    Mat gray = Mat::ones(rows,cols,CV_8UC1);
    for(int i=0;i<rows;i++){
        uchar* sdata = src.ptr<uchar>(i);
        uchar* gdata = gray.ptr<uchar>(i);
        for(int j=0;j<cols;j++){
            gdata[j]=sdata[3*j + channel];
        }
    }
    imshow("gray",gray);
    imshow("src",src);
    return gray;
}


void DWT(Mat& src){
    int cols = src.cols;
    int rows = src.rows;
    int count = 1;
    Mat tmp = Mat::ones(rows, cols, CV_32FC1);
    Mat dst = Mat::ones(rows, cols, CV_32FC1);
    Mat src_ = src.clone();
    src_.convertTo(src_, CV_32FC1);

    while (count<=depth){
        rows = src.rows / pow(2,count-1);
        cols = src.cols / pow(2,count-1);

        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols / 2; j++){
                tmp.at<float>(i, j) = (src_.at<float>(i, 2 * j) + src_.at<float>(i, 2 * j + 1)) / 2;
                tmp.at<float>(i, j + cols / 2) = (src_.at<float>(i, 2 * j) - src_.at<float>(i, 2 * j + 1)) / 2;
            }
        }
        for (int i = 0; i < rows / 2; i++){
            for (int j = 0; j < cols; j++){
                dst.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
                dst.at<float>(i + rows / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
            }
        }
        src_ = dst;
        count++;
    }

    dst.convertTo(dst, CV_8UC1);
    src = dst.clone();
    //imshow("DWT",src);
}


void CLAHE_LF(Mat &src){
    Rect rect = Rect(0,0, int(src.cols/pow(2,depth)),int(src.rows/pow(2,depth)));
    Mat LF = src(rect);
    Mat temp = Mat::zeros(LF.rows,LF.cols,CV_8UC1);
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->apply(LF,temp);
    temp.copyTo(src(rect));
    imshow("CLAHE_LF",src);
}

void reverse_DWT(Mat& src){
    int cols = src.cols;
    int rows = src.rows;
    int count = depth;
    Mat tmp = Mat::ones(rows, cols, CV_32FC1);
    Mat dst = Mat::ones(rows, cols, CV_32FC1);
    Mat src_ = src.clone();
    src_.convertTo(src_, CV_32FC1);

    while (count>0){
        rows = src.rows / pow(2,count-1);
        cols = src.cols / pow(2,count-1);

        for (int i = 0; i < rows/2; i++){
            for (int j = 0; j < cols; j++){
                tmp.at<float>(2 * i, j) = src_.at<float>(i, j) + src_.at<float>(i + rows/2, j);
                tmp.at<float>(2 * i +  1, j) = src_.at<float>(i, j) - src_.at<float>(i + rows/2, j);
            }
        }
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols/2; j++){
                dst.at<float>(i, 2 * j) = tmp.at<float>(i, j) + tmp.at<float>(i, j + cols/2);
                dst.at<float>(i, 2 * j + 1) = tmp.at<float>(i, j) - tmp.at<float>(i, j + cols/2);
            }
        }
        src_ = dst;
        count--;
    }

    dst.convertTo(dst, CV_8UC1);
    src = dst.clone();
    //imshow("reverse_DWT",src);
}



Mat get_H(Mat& src){
    float min = 255.0f;
    uchar max = 0.0f;
    int cols = src.cols;
    int rows = src.rows;

    Mat src_=src.clone();
    src_.convertTo(src_,CV_32FC1);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(src_.at<float>(i,j)>max)
                max = src.at<float>(i,j);
            if(src_.at<float>(i,j)<min)
                min = src.at<float>(i,j);
        }
    }

    Mat H = Mat::ones(src.rows,src.cols,CV_32FC1);
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            H.at<float>(i,j)=pow((src_.at<float>(i,j)-min)/(max-min), alpha);
        }
    }
    return H;
}

Mat weight_sum(Mat& gray, Mat& trans, Mat& H){
    int cols = gray.cols;
    int rows = gray.rows;

    Mat gray_=gray.clone();
    Mat trans_=trans.clone();
    Mat enhanced_ = Mat::ones(rows,cols,CV_32FC1);
    gray_.convertTo(gray_,CV_32FC1);
    trans.convertTo(trans_, CV_32FC1);


    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            enhanced_.at<float>(i,j)=
                    gray_.at<float>(i,j) * H.at<float>(i,j) +
                    beta * trans_.at<float>(i,j) * (1 - H.at<float>(i,j));
        }
    }
    Mat enhanced;
    enhanced_.convertTo(enhanced,CV_8UC1);
    //imshow("enhanced",enhanced);
    return enhanced;
}

Mat colouring(Mat& src, Mat& gray, Mat& enhanced){
    int cols = src.cols;
    int rows = src.rows;
    Mat src_=src.clone();
    Mat gray_ = gray.clone();
    Mat enhanced_=enhanced.clone();
    Mat colored_ = Mat::ones(rows,cols,CV_32FC3);
    src_.convertTo(src_,CV_32FC3);
    enhanced_.convertTo(enhanced_, CV_32FC1);
    gray_.convertTo(gray_,CV_32FC1);


    for(int i=0;i<rows;i++){
        float* sdata = src_.ptr<float>(i);
        float* edata = enhanced_.ptr<float>(i);
        float* gdata = gray_.ptr<float>(i);
        float* cdata = colored_.ptr<float>(i);
        for(int j=0;j<cols;j++){
            cdata[3*j]=sdata[3*j]*edata[j]/gdata[j];
            cdata[3*j+1]=sdata[3*j+1]*edata[j]/gdata[j];
            cdata[3*j+2]=sdata[3*j+2]*edata[j]/gdata[j];
        }
    }
    Mat colored;
    colored_.convertTo(colored,CV_8UC3);
    imshow("colored",colored);
    return colored;
}

void CLAHE_DWT_GRAY(Mat &gray){
    Mat trans = gray.clone();
    DWT(trans);
    CLAHE_LF(trans);
    reverse_DWT(trans);
    Mat H = get_H(gray);
    Mat enhanced = weight_sum(gray,trans,H);
    gray = enhanced.clone();
}

Mat union_mat(Mat& g0,Mat& g1, Mat& g2){
    Mat enhanced = Mat::ones(g0.rows, g0.cols, CV_8UC3);
    for(int i=0;i<g0.rows;i++){
        for(int j=0;j<g0.cols;j++){
            enhanced.at<Vec3b>(i,j)[0]=g0.at<uchar>(i,j);
            enhanced.at<Vec3b>(i,j)[1]=g1.at<uchar>(i,j);
            enhanced.at<Vec3b>(i,j)[2]=g2.at<uchar>(i,j);
        }
    }
    return enhanced;
}

Mat CLAHE_DWT(Mat &src){
    Mat gray0 = get_gray(src, 0);
    Mat gray1 = get_gray(src, 1);
    Mat gray2 = get_gray(src, 2);
    CLAHE_DWT_GRAY(gray0);
    CLAHE_DWT_GRAY(gray1);
    CLAHE_DWT_GRAY(gray2);


    Mat color_enhanced = union_mat(gray0, gray1, gray2);//colouring(src,gray,enhanced);
    return color_enhanced;
}


#endif //IMAGE_ENHANCE_CLAHE_DWT_H
