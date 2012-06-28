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

double BPFADictionaryLearner::betaRandom(boost::mt19937 &engine, const double &alpha, const double &beta)
{
	double x = boost::gamma_distribution<>(alpha, 1.0)(engine);
	double y = boost::gamma_distribution<>(beta, 1.0)(engine);
	return x / (x + y);
}

void BPFADictionaryLearner::init(const cv::Mat &_Y, const int &_K, const int &seed)
{
	this->K = _K;
	this->M = _Y.rows;
	this->N = _Y.cols;
	this->a = this->K;
	this->b = 1;

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

	pi.resize(K);
	cout << b*(K-1)/K << endl;
	for(int k=0; k<K; ++k){
		pi[k] = betaRandom(engine, a/K, b*(K-1)/K);
	};
	boost::copy(pi, ostream_iterator<double>(std::cout, " "));
	cout << "K: " << K << endl
		 << "M: " << M << endl
		 << "N: " << N << endl;
}


void BPFADictionaryLearner::train(const int iteration)
{
	for(int t=0; t<iteration; ++t)
	{
	}
	cout << "sample D" << endl; sampleD(); 
	cout << "sample Z" << endl; sampleZ();
	cout << "sample S" << endl; sampleS();
	cout << "sample Pi" << endl; samplePi();
	cout << "sample Gamma_s" << endl; sampleGamma_s();
	cout << "sample Gamma_e" << endl; sampleGamma_e();
}

void BPFADictionaryLearner::sampleD(void)
{
	cv::Mat X;
	cv::multiply(Z, S, X);
	cv::Mat E = Y - D * X;

	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k = D.col(k);
		double x_k_square = cv::Mat(x_k * x_k.t()).at<double>(0, 0);
		double variance = 1.0 / (M + gamma_e * x_k_square);
		int N_nonzero = cv::countNonZero(x_k);
		
		vector<int> omega;
		omega.reserve(N_nonzero);
		for(int i=0; i<N; ++i){
			if(x_k.at<double>(0, i) != 0.0){
				omega.push_back(i);
			}
		}
		
		cv::Mat E_k = E + d_k * x_k;
		cv::Mat mean = (gamma_e * variance) * (E_k * x_k.t());
//		cv::Mat mean = (gamma_e * variance) * (E * x_k.t() + d_k * x_k_square);

		double stddev = sqrt(variance);
		cv::Mat sample(d_k.size(), d_k.type());
		for(int m=0; m<M; ++m){
			double d_km = boost::normal_distribution<>(mean.at<double>(m, 0), stddev)(engine);
			sample.at<double>(m, 0) = d_km;
		}
		sample.copyTo(d_k);

		E = E_k - d_k * x_k;
	}
}



void BPFADictionaryLearner::sampleZ(void)
{
	cv::Mat X;
	cv::multiply(Z, S, X);
	cv::Mat E = Y - D * X;

	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k = D.col(k);
		double d_k_square = cv::Mat(d_k.t() * d_k).at<double>(0, 0);
		cv::Mat E_k = E + d_k * x_k;

		for(int i=0; i<N; ++i){
			double s_ik = S.at<double>(k, i);
			double dkTEki = cv::Mat(d_k.t() * E_k.col(i)).at<double>(0, 0);
			double p1 = pi[k] * exp(-gamma_e / 2 * (s_ik * s_ik * d_k_square - 2 * s_ik * dkTEki));
			double p0 = 1 - pi[k];
			int z_ik = boost::bernoulli_distribution<>(p1 / (p0 + p1))(engine);
			Z.at<double>(k, i) = z_ik;
		}
	}
}



void BPFADictionaryLearner::sampleS(void)
{
}
void BPFADictionaryLearner::samplePi(void)
{
}
void BPFADictionaryLearner::sampleGamma_s(void)
{
}
void BPFADictionaryLearner::sampleGamma_e(void)
{
}