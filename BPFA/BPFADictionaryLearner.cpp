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
	E = Y - D * X;
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
	//E = Y - D * X;

	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k_old = D.col(k);
		cv::Mat z_k = Z.row(k);
		double x_k_square = x_k.dot(x_k);
		double variance = 1.0 / (M + gamma_e * x_k_square);
		
		cv::Mat Exk_dkxkxk = cv::Mat::zeros(M, 1, E.type());
		for(int i=0; i<N; ++i){
			if(z_k.at<double>(0, i) == 1.0){
				double x_ki = x_k.at<double>(0, i);
				Exk_dkxkxk += (E.col(i) + d_k_old * x_ki) * x_ki;
			}
		}
//		cv::Mat mean = (gamma_e * variance) * (E * x_k.t() + d_k * x_k_square);
		cv::Mat mean = (gamma_e * variance) * Exk_dkxkxk;


		double stddev = sqrt(variance);
		cv::Mat d_k_new(d_k_old.size(), d_k_old.type());
		for(int m=0; m<M; ++m){
			double d_km = boost::normal_distribution<>(mean.at<double>(m, 0), stddev)(engine);
			d_k_new.at<double>(m, 0) = d_km;
		}
		
		for(int i=0; i<N; ++i){
			if(z_k.at<double>(0, i) == 1.0){
				E.col(i) += (d_k_old - d_k_new) * x_k.at<double>(0, i);
			}
		}
		d_k_new.copyTo(d_k_old);
	}
}



void BPFADictionaryLearner::sampleZ(void)
{
	cv::multiply(Z, S, X);
	//cv::Mat E = Y - D * X;

	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k = D.col(k);
		cv::Mat z_k_old = Z.row(k);
		double d_k_square = d_k.dot(d_k);
		cv::Mat dkTE = d_k.t() * E;
		
		cv::Mat z_k_new(x_k.size(), x_k.type());
		for(int i=0; i<N; ++i){
			double s_ik = S.at<double>(k, i);
			double d_kTEk_i = dkTE.at<double>(0, i);
			double x_ki = x_k.at<double>(0, i);
			if(z_k_old.at<double>(0, i) == 1.0){
				d_kTEk_i += x_ki * d_k_square;
			}

			int z_ki;
			double exponent = -gamma_e / 2 * (s_ik * s_ik * d_k_square - 2 * s_ik * d_kTEk_i);
			if(exponent > 650.0){ // In this case p will be almost 1. 
				z_ki = 1;
			}
			else if(exponent < -650.0){ // the case p will be almost 0
				z_ki = 0;
			}
			else{
				double p1 = pi[k] * exp(exponent);
				double p0 = 1 - pi[k];
				double p = p1 / (p0 + p1);
				z_ki = boost::bernoulli_distribution<>(p)(engine);
			}
			z_k_new.at<double>(0, i) = z_ki;
		}
		cv::Mat x_k_new;
		cv::multiply(z_k_new, S.row(k), x_k_new);

		//E = E_k - d_k * x_k_new;
		for(int i=0; i<N; ++i){
			if(z_k_old.at<double>(0, i) == 1.0 || z_k_new.at<double>(0, i) == 1.0){
				double x_ki = x_k.at<double>(0, i) - x_k_new.at<double>(0, i);
				E.col(i) += x_ki * d_k;
			}
		}
		z_k_new.copyTo(Z.row(k));
		x_k_new.copyTo(X.row(k));
	}
}



void BPFADictionaryLearner::sampleS(void)
{
	cv::multiply(Z, S, X);
	//cv::Mat E = Y - D * X;
	
	for(int k=0; k<K; ++k){
		cv::Mat x_k = X.row(k);
		cv::Mat d_k = D.col(k);
		cv::Mat z_k = Z.row(k);
		double d_k_square = d_k.dot(d_k);
		cv::Mat dkTE = d_k.t() * E;

		double variance0 = 1.0 / gamma_s;
		double variance1 = 1.0 / (gamma_s + gamma_e * d_k_square);
		
		cv::Mat s_k_new(x_k.size(), x_k.type());
		for(int i=0; i<N; ++i){
			double variance;
			double mean;
			if(z_k.at<double>(0, i) == 1.0){
				variance = variance1;
				//double d_kTEk_i = d_k.dot(E.col(i) + d_k * x_k.at<double>(0, i));
				double d_kTEk_i = dkTE.at<double>(0, i) + x_k.at<double>(0, i) * d_k_square;
				mean = gamma_e * variance * d_kTEk_i;
			}
			// z_ik == 0
			else{
				variance = variance0;
				mean = 0;
			}

			double s_ki = boost::normal_distribution<>(mean, sqrt(variance))(engine);
			s_k_new.at<double>(0, i) = s_ki;
		}
		s_k_new.copyTo(S.row(k));
		cv::Mat x_k_new;
		cv::multiply(s_k_new, Z.row(k), x_k_new);
		
		//E += d_k * (x_k - x_k_new);
		for(int i=0; i<N; ++i){
			if(z_k.at<double>(0, i) == 1.0){
				double x_ki = x_k.at<double>(0, i) - x_k_new.at<double>(0, i);
				E.col(i) += x_ki * d_k;
			}
		}
		x_k_new.copyTo(X.row(k));
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
	//cv::Mat E = Y - D * X;
	double e_mn_sqsum = E.dot(E);
	double shape = e + M * N / 2.0;
	double scale = 1.0 / (f + e_mn_sqsum / 2.0);
	//cout << "\tshape_e: " << shape << ", scale_e: " << scale << endl;
	boost::math::gamma_distribution<> dist(shape, scale);
	gamma_e = boost::math::quantile(dist, boost::uniform_01<>()(engine));
}