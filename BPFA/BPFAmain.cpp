// BPFA.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "BPFADictionaryLearner.h"

using namespace std;
using namespace boost;
using namespace cv;


cv::Mat imageToPatches(const cv::Mat &image, const int &patch_size)
{
	int w = image.cols;
	int h = image.rows;
	int l = patch_size;
	int wp = w / l;	// number of patches in a horizontal direction
	int hp = h / l;	// number of patches in a vertical direction
	int num_patches = wp * hp;
	cv::Mat patches(l*l, num_patches, CV_64FC1);

	for(int y=0; y<hp; ++y){
		for(int x=0; x<wp; ++x){ 
			cv::Rect roi(x*l, y*l, l, l);
			cv::Mat patch = image(roi).clone();
			cv::Mat column_vectorized_image = patch.reshape(0, 1).t();
			column_vectorized_image.copyTo(patches.col(wp*y+x));
		}
	}

	return patches;
}









cv::Mat imageToDensePatches(const cv::Mat &image, const int &patch_size)
{
	int w = image.cols;
	int h = image.rows;
	int l = patch_size;
	int wp = w - l + 1;	// number of patches in a horizontal direction
	int hp = h - l + 1;	// number of patches in a vertical direction
	int num_patches = wp * hp;
	cv::Mat patches(l*l, num_patches, CV_64FC1);

	for(int y=0; y<hp; ++y){
		for(int x=0; x<wp; ++x){
			cv::Rect roi(x, y, l, l);
			cv::Mat patch = image(roi).clone();
			cv::Mat column_vectorized_image = patch.reshape(0, 1).t();
			column_vectorized_image.copyTo(patches.col(wp*y+x));
		}
	}

	return patches;
}









cv::Mat colorImageToPatches(const cv::Mat &image, const int &patch_size)
{
	int w = image.cols;
	int h = image.rows;
	int l = patch_size;
	int wp = w / l;	// number of patches in a horizontal direction
	int hp = h / l;	// number of patches in a vertical direction
	int num_patches = wp * hp;
	vector<cv::Mat> channels;
	cv::split(image, channels);
	cv::Mat patches(l*l*3, num_patches, CV_64FC1);


	for(int c=0; c<3; ++c){
		for(int y=0; y<hp; ++y){
			for(int x=0; x<wp; ++x){
				cv::Rect roi(x*l, y*l, l, l);
				cv::Mat patch = channels[c](roi).clone();
				cv::Mat column_vectorized_image = patch.reshape(0, 1).t();
				column_vectorized_image.copyTo(patches(Rect(wp*y+x, c*l*l, 1, l*l)));
			}
		}
	}
	return patches;
}









cv::Mat patchesToImage(const cv::Mat &patches, const int &width)
{
	int patch_size = sqrt(static_cast<double>(patches.rows));
	int num_patches = patches.cols;
	int l = patch_size;
	int wp = width / l;			// number of patches in a horizontal direction
	int hp = num_patches / wp;	// number of patches in a vertical direction
	int w = width;
	int h = hp * l;
	cv::Mat image(h, w, CV_64FC1);

	for(int i=0; i<num_patches; ++i){
		cv::Rect roi((i%wp)*l, (i/wp)*l, l, l);
		cv::Mat column_vectorized_image = patches.col(i).clone();
		column_vectorized_image.reshape(0, l).copyTo(image(roi));
	}

	return image;
}














cv::Mat densePatchesToImage(const cv::Mat &patches, const int &width)
{
	int patch_size = sqrt(static_cast<double>(patches.rows));
	int num_patches = patches.cols;
	int l = patch_size;
	int wp = width - l + 1;		// number of patches in a horizontal direction
	int hp = num_patches / wp;	// number of patches in a vertical direction
	int w = width;
	int h = hp + l - 1;
	cv::Mat patch_aggregation_image(h, w, CV_64FC1);
	cv::Mat one = Mat::ones(l, l, CV_64FC1);
	cv::Mat counts(h, w, CV_64FC1);

	for(int y=0; y<hp; ++y){
		for(int x=0; x<wp; ++x){
			cv::Rect roi(x, y, l, l);
			cv::Mat patch;
			patches.col(wp*y+x).convertTo(patch, CV_64FC1);
			patch_aggregation_image(roi) += patch.reshape(0, l);
			
			counts(roi) += one.reshape(0, l);
		}
	}

	cv::Mat average(h, w, CV_64FC1);
	cv::divide(patch_aggregation_image, counts, average);
	cv::Mat image(h, w, CV_64FC1);
	average.convertTo(image, CV_64FC1);

	return image;
}









cv::Mat colorPatchesToImage(const cv::Mat &patches, const int &width)
{
	int patch_size = sqrt(static_cast<double>(patches.rows)/3);
	int num_patches = patches.cols;
	int l = patch_size;
	int wp = width / l;			// number of patches in a horizontal direction
	int hp = num_patches / wp;	// number of patches in a vertical direction
	int w = width;
	int h = hp * l;
	cv::Mat image(h, w, CV_64FC3);
	
	vector<cv::Mat> channels(3);
	for(int c=0; c<3; ++c){
		channels[c] = cv::Mat(h, w, CV_64FC1);
		for(int i=0; i<num_patches; ++i){
			int x = i % wp;
			int y = i / wp;
			cv::Rect roi(x*l, y*l, l, l);
			cv::Mat patch = patches.col(i)(Rect(0, c*l*l, 1, l*l)).clone();
			patch.reshape(0, l).copyTo(channels[c](roi));
		}
	}

	cv::merge(channels, image);

	return image;
}









void runKSVDForGrayscaleImage(const string &filename)
{
	cv::Mat image = cv::imread(filename, 0);
	image.convertTo(image, CV_64FC1, 1.0/256.0);
	cv::Mat noise(image.rows, image.cols, CV_64FC1);
	cv::randn(noise, cv::Scalar(0), cv::Scalar(10));
	//image += noise / 256.0;
	const int patch_size = 8;
	cv::Mat patches = imageToPatches(image, patch_size);
	imshow("original", image);
	cv::waitKey(1);

	const int K = 256;
	const int n = patches.cols; //patch_size * patch_size
	cv::Mat Y = patches;
	BPFADictionaryLearner dictionaryLearner;
	cout << "init" << endl;
	dictionaryLearner.init(Y, K);

	for(int i=0; i<200; ++i){
		cout << endl <<  "round " << i << endl;
		dictionaryLearner.train(1);

		cv::Mat resultPatches = dictionaryLearner.D * dictionaryLearner.X;
		imshow("sc", patchesToImage(resultPatches, image.cols));

		cv::Mat resultDictionary = dictionaryLearner.D + 0.5;
		imshow("D", patchesToImage(resultDictionary, patch_size*sqrt(static_cast<double>(K))));
		waitKey(1);
	}
}









void runKSVDForColorImage(const string &filename)
{
	cv::Mat image = cv::imread(filename, 1);
	image.convertTo(image, CV_64FC3, 1.0/256.0);
	cv::Mat noise(image.rows, image.cols, CV_64FC3);
	cv::randn(noise, cv::Scalar(0), cv::Scalar(10));
	//image += noise / 64.0;
	const int patch_size = 8;
	cv::Mat patches = colorImageToPatches(image, patch_size);
	imshow("original", image);
	cv::waitKey(1);

	const int K = 256;
	const int n = patches.rows; //patch_size * patch_size
	cv::Mat Y = patches;
	BPFADictionaryLearner dictionaryLearner;
	cout << "init" << endl;
	dictionaryLearner.init(Y, K);

	for(int i=0; i<200; ++i){
		cout << endl <<  "round " << i << endl;
		dictionaryLearner.train(1);

		cv::Mat resultPatches = dictionaryLearner.D * dictionaryLearner.X;
		imshow("sc", colorPatchesToImage(resultPatches, image.cols));
		waitKey(10);

		cv::Mat resultDictionary = dictionaryLearner.D + 0.5;
		imshow("D", colorPatchesToImage(resultDictionary, patch_size*sqrt(static_cast<double>(K))));
		waitKey(10);
	}
}









void runKSVDDenoiseGrayscaleImage(const string &filename)
{
	cv::Mat image = cv::imread(filename, 0);
	image.convertTo(image, CV_64FC1, 1.0/256.0);
	cv::Mat noise(image.rows, image.cols, CV_64FC1);
	cv::randn(noise, cv::Scalar(0), cv::Scalar(10));
	cv::Mat noisyImage = image + noise / 256.0;
	const int patch_size = 8;
	cv::Mat patches = imageToDensePatches(noisyImage, patch_size);
	imshow("original", image);
	imshow("noisyImage", noisyImage);
	cv::waitKey(1);

	const int K = 256;
	const int n = patches.rows; //patch_size * patch_size
	cv::Mat Y = patches;
	BPFADictionaryLearner dictionaryLearner;
	cout << "init" << endl;
	dictionaryLearner.init(Y, K);

	for(int i=0; i<200; ++i){
		cout << endl <<  "round " << i << endl;
		dictionaryLearner.train(1);

		cv::Mat resultPatches = dictionaryLearner.D * dictionaryLearner.X;
		imshow("sc", densePatchesToImage(resultPatches, image.cols));

		cv::Mat resultDictionary = dictionaryLearner.D + 0.5;
		imshow("D", patchesToImage(resultDictionary, patch_size*sqrt(static_cast<double>(K))));
		waitKey(1);
	}
}









int _tmain(int argc, _TCHAR* argv[])
{
	const string filename = "barbara.jpg";
	//const string filename = "castle.png";

	runKSVDForGrayscaleImage(filename);
	//runKSVDForColorImage(filename); 
	//runKSVDDenoiseGrayscaleImage(filename);

	waitKey();
	return 0;
}

