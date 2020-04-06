//OpenCv边缘检测：单线程与OpenMP多线程对比
#include<iostream>
#include<omp.h>
#include<time.h>
#include<math.h>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>

#define N 200//边缘检测次数,增加运算量

clock_t start,stop;

using namespace std;
using namespace cv;

//边缘检测函数定义
//sour:源图像,des:目标图像
//thresh:像素阈值,相邻两像素的差值大于此阈值则判定为边缘
//单线程边缘检测
int EdgeDetection(Mat &sour,Mat &des,int thresh)
{
	if(sour.isContinuous())//判断图像在内存中是否连续储存，若是则返回True
	{
		//图像连续储存，则可以看成一个一维数组进行运算
		Mat copysour=sour.clone();//拷贝源图像，保护原始数据
		int height=sour.rows;//图像高度等于行数
		int width=sour.cols;//图像宽度等于列数
		int step=sour.step;//图像每一行所占的字节数

		uchar* sourdata=copysour.data;//源图像无符号字符指针
		uchar* desdata=des.data;//目标图像无符号字符指针

		for(int i=step+1;i<height*width;i++)//遍历图像一维数组,跳过图像上边缘像素点
		{
			if(i%step==0)//指针指向图片左边缘像素点
				i++;//跳过图像左边缘像素点
			if(abs(sourdata[i]-sourdata[i-1])>thresh//中心像素点和左像素点对比
				||abs(sourdata[i]-sourdata[i-step])>thresh//中心像素点和上像素点对比
				||abs(sourdata[i]-sourdata[i-step-1])>thresh)//中心像素点和左上像素点对比
				desdata[i]=255;//像素值差值大于阈值，判定为边缘，设置为白色
			else
				desdata[i]=0;//否则为整体背景，设置为黑色
		}
		return 0;//执行成功返回
	}
	else
		return -1;//执行失败返回
}

//OMP多线程边缘检测
int EdgeDetectionOMP(Mat &sour,Mat &des,int thresh)
{
	if(sour.isContinuous())//判断图像在内存中是否连续储存，若是则返回True
	{
		//图像连续储存，则可以看成一个一维数组进行运算
		Mat copysour=sour.clone();//拷贝源图像，保护原始数据
		int height=sour.rows;//图像高度等于行数
		int width=sour.cols;//图像宽度等于列数
		int step=sour.step;//图像每一行所占的字节数

		uchar* sourdata=copysour.data;//源图像无符号字符指针
		uchar* desdata=des.data;//目标图像无符号字符指针
#pragma omp parallel for num_threads(8)//使用OpenMP指导语句进行加速
		for(int i=step+1;i<height*width;i++)//遍历图像一维数组,跳过图像上边缘像素点
		{
			if(i%step==0)//指针指向图片左边缘像素点
				continue;//跳过图像左边缘像素点
			if(abs(sourdata[i]-sourdata[i-1])>thresh//中心像素点和左像素点对比
				||abs(sourdata[i]-sourdata[i-step])>thresh//中心像素点和上像素点对比
				||abs(sourdata[i]-sourdata[i-step-1])>thresh)//中心像素点和左上像素点对比
				desdata[i]=255;//像素值差值大于阈值，判定为边缘，设置为白色
			else
				desdata[i]=0;//否则为整体背景，设置为黑色
		}
		return 0;//执行成功返回
	}
	else
		return -1;//执行失败返回
}

int main()
{
	Mat img=imread("scenery.jpg");//相对路径读入图片
	Mat grayimg;//定义灰度图像

	//显示彩色图像
	namedWindow("rgbimage",CV_WINDOW_AUTOSIZE);
	imshow("rgbimage",img);
	waitKey();//等待按键

	cvtColor(img,grayimg,CV_RGB2GRAY);//RGB图转灰度图

	//显示灰度图像
	namedWindow("grayimage",CV_WINDOW_AUTOSIZE);
	imshow("grayimage",grayimg);
	waitKey();

	Mat edgeimg=grayimg.clone();//定义边缘图像并赋初值

	//单线程
	start=clock();
	for(int i=0;i<N;i++)
	{
		int flag=EdgeDetection(grayimg,edgeimg,40);//进行边缘检测
		if(flag)
			break;//执行失败则跳出
	}
	stop=clock();

	//显示边缘图像
	namedWindow("edgeimage1",CV_WINDOW_AUTOSIZE);
	imshow("edgeimage1",edgeimg);

	cout<<"single thread, execute time: "<<
		(double)(stop-start)*1000/CLOCKS_PER_SEC<<"ms"<<endl;
	waitKey();

	//OpenMP多线程
	start=clock();
	for(int i=0;i<N;i++)
	{
		int flag=EdgeDetectionOMP(grayimg,edgeimg,40);
		if(flag)
			break;
	}
	stop=clock();

	//显示边缘图像
	namedWindow("edgeimage2",CV_WINDOW_AUTOSIZE);
	imshow("edgeimage2",edgeimg);

	cout<<"multiple threads, execute time: "<<
		(double)(stop-start)*1000/CLOCKS_PER_SEC<<"ms"<<endl;
	
	waitKey();
	destroyAllWindows();//关闭所有窗口
	return 0;
}

