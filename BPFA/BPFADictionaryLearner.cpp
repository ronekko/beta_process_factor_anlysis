// KSVD.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "BPFADictionaryLearner.h"
#include <omp.h>

using namespace std;
using namespace boost;
using namespace cv;

BPFADictionaryLearner::BPFADictionaryLearner(){}
BPFADictionaryLearner::~BPFADictionaryLearner(){} 

void BPFADictionaryLearner::init(const cv::Mat &_Y, const int &_K, const int &seed)
{
	this->K = _K;
	this->M = _Y.rows;
	this->N = _Y.cols;

	Y = cv::Mat(M, N, CV_64FC1);
	_Y.convertTo(Y, CV_64FC1);
	D = cv::Mat(M, K, CV_64FC1);
	X = cv::Mat(K, N, CV_64FC1);
	Z = cv::Mat::zeros(K, N, CV_64FC1);
	S = cv::Mat::zeros(K, N, CV_64FC1);
	engine.seed(seed);
	
	cv::randu(D, cv::Scalar(0), cv::Scalar(256));
	for(int j=0; j<D.cols; ++j){
		cv::Mat col = D.col(j);
		double l2norm = cv::norm(col, NORM_L2);
		D.col(j) /= l2norm;
	}

	cout << "K: " << K << endl
		 << "M: " << M << endl
		 << "N: " << N << endl;
}


void BPFADictionaryLearner::train(const int iteration)
{
	for(int t=0; t<iteration; ++t)
	{
	}	
}
