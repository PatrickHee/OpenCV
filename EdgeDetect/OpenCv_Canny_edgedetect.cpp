//OpenCv——Canny图片边缘检测算法
#include<iostream>
#include<time.h>
#include<omp.h>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat img=imread("scenery.jpg");//读入图像
	namedWindow("rgbimage");//创建图像显示窗口
	imshow("rgbimage",img);//显示RGB图像
	waitKey();//等待按键

	Mat grayimg;
	cvtColor(img,grayimg,CV_RGB2GRAY);//RGB图像转灰度图
	namedWindow("grayimage");
	imshow("grayimage",grayimg);
	waitKey();

	Mat edgeimg;
	Canny(grayimg,edgeimg,100,200);//采用Canny算法提取图像边缘
	namedWindow("edgeimage");
	imshow("edgeimage",edgeimg);
	waitKey();
	destroyAllWindows();

	return 0;
}
