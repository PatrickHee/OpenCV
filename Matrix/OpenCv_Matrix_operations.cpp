#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
	//矩阵定义并初始化
	uchar a[]={1,0,0,1,1,0,1,1,1};
	Mat M0(Size(3,3),CV_8UC1,a);
	cout<<"M0="<<endl<<M0<<endl<<endl;

	Mat M1(Size(4,4),CV_8UC1);//定义4*4的单通道无符号矩阵M1，元素随机
	cout<<"M1="<<endl<<M1<<endl<<endl;

	Mat M2(Size(4,4),CV_8UC3);//定义4*4的三通道无符号矩阵M2，元素随机
	cout<<"M2="<<endl<<M2<<endl<<endl;
	
	Mat Mzero=Mat::zeros(Size(4,4),CV_8UC1);//定义4*4的无符号单通道矩阵Mzero，元素全0
	cout<<"Mzero="<<endl<<Mzero<<endl<<endl;

	Mat Mone=Mat::ones(Size(4,4),CV_8UC1);//定义4*4的无符号单通道矩阵Mone，元素全1
	cout<<"Mone="<<endl<<Mone<<endl<<endl;

	//矩阵运算
	//矩阵相加
	Mat Madd=Mone+Mone;
	cout<<"Madd="<<endl<<Madd<<endl<<endl;

	//矩阵相减
	Mat Mszero=Mat::zeros(Size(4,4),CV_8SC1);
	Mat Msone=Mat::ones(Size(4,4),CV_8SC1);
	Mat Mminus=Mszero-Msone;
	cout<<"Mminus="<<endl<<Mminus<<endl<<endl;

	//矩阵转置
	Mat Mtrans=M0.t();
	cout<<"Mtrans="<<endl<<Mtrans<<endl<<endl;

	//矩阵行列式求值
	float b[]={1,2,3,4};
	Mat Mdet(Size(2,2),CV_32FC1,b);
	double det=determinant(Mdet);
	cout<<"Mdet="<<endl<<Mdet<<endl<<endl<<"det(Mdet)="<<det<<endl;
	return 0;
}
