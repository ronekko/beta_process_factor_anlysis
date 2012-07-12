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
	boost::math::beta_distribution<> dist(alpha, beta);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}

void BPFADictionaryLearner::init(const cv::Mat &_Y, const int &_K, const int &seed)
{
	this->K = _K;
	this->M = _Y.rows;
	this->N = _Y.cols;
	this->a = this->K;
	this->b = 1;
	this->c = 1e-6;
	this->d = 1e-6;
	this->e = 1e-6;
	this->f = 1e-6;

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
	for(int k=0; k<K; ++k){
		pi[k] = betaRandom(engine, a/K, b*(K-1)/K);
	};
	
//	gamma_s = boost::gamma_distribution<>(c, 1.0/d)(engine);
//	gamma_e = boost::gamma_distribution<>(e, 1.0/f)(engine);
	gamma_s = 1.0;
	gamma_e = 1.0;

	cout << "K: " << K << endl
		 << "M: " << M << endl
		 << "N: " << N << endl
		 << "gamma_s: " << gamma_s << endl
		 << "gamma_e: " << gamma_e << endl;
}


void BPFADictionaryLearner::train(const int iteration)
{
	for(int t=0; t<iteration; ++t)
	{
		cout << "sample ";
		sampleD();	cout << "D ";
		samplePi();	cout << "Pi ";
		sampleZ();	cout << "Z ";
		sampleS();	cout << "S ";
		cout << endl;
		sampleGamma_s(); cout << "sample Gamma_s: " << gamma_s << endl;
		sampleGamma_e(); cout << "sample Gamma_e: " << gamma_e << endl;
	}
}

void BPFADictionaryLearner::sampleD(void)
{
	cv::multiply(Z, S, X);
	cv::Mat E = Y - D * X;

	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k = D.col(k);
		double x_k_square = x_k.dot(x_k);
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
	cv::multiply(Z, S, X);
	cv::Mat E = Y - D * X;

	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k = D.col(k);
		double d_k_square = d_k.dot(d_k);
		cv::Mat E_k = E + d_k * x_k;
		cv::Mat dkTEk = d_k.t() * E_k;
		
		cv::Mat sample(x_k.size(), x_k.type());
		for(int i=0; i<N; ++i){
			double s_ik = S.at<double>(k, i);
			double dkTEki = dkTEk.at<double>(0, i);
			int z_ki;
			double exponent = -gamma_e / 2 * (s_ik * s_ik * d_k_square - 2 * s_ik * dkTEki);
			if(exponent > 700.0){ // In this case p will be almost 1. 
				z_ki = 1;
			}
			else if(exponent < -700.0){ // the case p will be almost 0
				z_ki = 0;
			}
			else{
				double p1 = pi[k] * exp(exponent);
				double p0 = 1 - pi[k];
				double p = p1 / (p0 + p1);
				z_ki = boost::bernoulli_distribution<>(p)(engine);
			}
			sample.at<double>(0, i) = z_ki;
		}
		sample.copyTo(Z.row(k));
		cv::Mat x_k_new;
		cv::multiply(sample, S.row(k), x_k_new);
		x_k_new.copyTo(X.row(k));

		E = E_k - d_k * x_k_new;
	}
}



void BPFADictionaryLearner::sampleS(void)
{
	cv::multiply(Z, S, X);
	cv::Mat E = Y - D * X;
	
	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k = D.col(k);
		cv::Mat z_k = Z.row(k);
		cv::Mat E_k = E + d_k * x_k;
		cv::Mat dkTEk = d_k.t() * E_k;
		double d_k_square = d_k.dot(d_k);

		double variance0 = 1.0 / gamma_s;
		double variance1 = 1.0 / (gamma_s + gamma_e * d_k_square);
		
		cv::Mat sample(x_k.size(), x_k.type());
		for(int i=0; i<N; ++i){
			double variance;
			double mean;
			if(z_k.at<double>(0, i) == 1.0){
				variance = variance1;
				mean = gamma_e * variance * dkTEk.at<double>(0, i);
			}
			// z_ik == 0
			else{
				variance = variance0;
				mean = 0;
			}

			double s_ki = boost::normal_distribution<>(mean, sqrt(variance))(engine);
			sample.at<double>(0, i) = s_ki;
		}
		sample.copyTo(S.row(k));
		cv::Mat x_k_new;
		cv::multiply(sample, Z.row(k), x_k_new);
		x_k_new.copyTo(X.row(k));

		E = E_k - d_k * x_k_new;
	}
}



void BPFADictionaryLearner::samplePi(void)
{
	for(int k=0; k<K; ++k){
		int num_ones = cv::countNonZero(Z.row(k));
		pi[k] = betaRandom(engine, a/K + num_ones, b*(K-1)/K + N - num_ones);
	}
}



void BPFADictionaryLearner::sampleGamma_s(void)
{
	double s_ki_sqsum = S.dot(S); // \sum_i \sum_k S_{ik}^2
	double shape = c + K * N / 2.0;
	double scale = 1.0 / (d + s_ki_sqsum / 2.0);
	//cout << "\tshape_s: " << shape << ", scale_s: " << scale << endl;
	boost::math::gamma_distribution<> dist(shape, scale);
	gamma_s = boost::math::quantile(dist, boost::uniform_01<>()(engine));
}



void BPFADictionaryLearner::sampleGamma_e(void)
{
	cv::Mat E = Y - D * X;
	double e_mn_sqsum = E.dot(E);
	double shape = e + M * N / 2.0;
	double scale = 1.0 / (f + e_mn_sqsum / 2.0);
	//cout << "\tshape_e: " << shape << ", scale_e: " << scale << endl;
	boost::math::gamma_distribution<> dist(shape, scale);
	gamma_e = boost::math::quantile(dist, boost::uniform_01<>()(engine));
}