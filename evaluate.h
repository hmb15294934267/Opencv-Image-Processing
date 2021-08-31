#pragma once

#include<vector>
#include<string>
#include<iostream>

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

using namespace std;
using namespace cv;


const double threshold = 1.5;			//特征点检测评估阈值

class Evaluate
{
public:

	Evaluate() {}

	~Evaluate(){}

	/*删除矩阵中指定列*/
	void delete_col(Mat& object, int num);

	void delete_nums_cols(Mat& object, Mat& arrs, vector<int>& nums, int n);

	/*删除不合格点后的特征点数*/
	int delete_unqualified_points(const Mat& image, const Mat& homographyconst,
		string model, vector<KeyPoint> keys, vector<int>& nums, int N);

	/*特征点检测评价*/
	float detect_evaluation(const Mat& image_1, const Mat& image_2, const Mat& homographyconst,
		string model, vector<KeyPoint> keys_1, vector<KeyPoint> keys_2);

	/*特征点匹配评价*/
	int match_evaluation();
};
