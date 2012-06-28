#pragma once

#include "stdafx.h"

// Mingyuan Zhou, Haojun Chen, John Paisley, Lu Ren, Lingbo Li, Zhengming Xing, David Dunson, Guillermo Sapiro and Lawrence Carin,
// "Nonparametric Bayesian Dictionary Learning for Analysis of Noisy and Incomplete Images,",
// IEEE Trans. Image Processing, Vol. 21, pp. 130-144, Jan. 2012.
class BPFADictionaryLearner
{
public:
	cv::Mat Y;
	cv::Mat D;
	cv::Mat Z;
	cv::Mat S;
	std::vector<double> pi;
	double gamma_e;
	double gamma_s;
	int K;		// number of dictionary atoms
	int M;		// dimension of data
	int N;		// number of samples
	cv::Mat X;
	boost::random::mt19937 engine;

	BPFADictionaryLearner();
	~BPFADictionaryLearner();
	void init(const cv::Mat &Y, const int &K, const int &seed = 0);
	void train(const int R); // R: number of iteration
};