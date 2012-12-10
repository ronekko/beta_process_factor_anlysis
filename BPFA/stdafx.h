// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: プログラムに必要な追加ヘッダーをここで参照してください。
#ifdef _DEBUG
	#pragma comment(lib, "opencv_core243d.lib")
	#pragma comment(lib, "opencv_imgproc243d.lib")
	#pragma comment(lib, "opencv_highgui243d.lib")
	#pragma comment(lib, "opencv_features2d243d.lib")
#else
	#pragma comment(lib, "opencv_core243.lib")
	#pragma comment(lib, "opencv_imgproc243.lib")
	#pragma comment(lib, "opencv_highgui243.lib")
	#pragma comment(lib, "opencv_features2d243.lib")
#endif


#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <set>
#include <map>
#include <utility>
#include <memory>
#include <typeinfo>

#include "direct.h"

#include <boost/unordered_set.hpp>
#include <boost/unordered_map.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>
#include <boost/timer.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/random.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/program_options.hpp>
#include <boost/chrono.hpp>

#include <opencv2/opencv.hpp>