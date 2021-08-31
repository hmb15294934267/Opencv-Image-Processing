# include"evaluate.h"
# include<vector>

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

#include<vector>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;


/*******************删除矩阵中指定的列************************/
void Evaluate::delete_col(Mat& object, int num)
{
	if (num < 0 || num >= object.cols)
	{
		cout << " 列标号不在矩阵正常范围内 " << endl;
	}
	else
	{
		//删除列是矩阵的最后一列
		if (num == object.cols - 1)
		{
			object = object.t();			 //求逆矩阵
			object.pop_back();				 //弹出最后一行元素
			object = object.t();
		}
		else
		{
			//num 列之后的所有数据前移一列
			for (int i = num + 1; i < object.cols; i++)
			{
				object.col(i - 1) = object.col(i) + Scalar(0, 0, 0, 0);
			}
			object = object.t();			
			object.pop_back();				
			object = object.t();
		}
	}
}


/*******************删除矩阵中多个指定列************************/
/*object 需要进行操作的矩阵
 *arrs 删除不合格特征点后的矩阵*/
void Evaluate::delete_nums_cols(Mat& object, Mat& arrs, vector<int>& nums, int n)
{
	arrs.create(2, n, CV_32FC1);

	//获取矩阵每一行的首地址
	float* p10 = arrs.ptr<float>(0), * p11 = arrs.ptr<float>(1);
	float* p20 = object.ptr<float>(0), * p21 = object.ptr<float>(1);

	for (int i = 0; i < object.cols; i++)
	{
		auto it = nums.begin();
		for (; it != nums.end(); it++)
		{
			if ((*it) < 0 || (*it) >= object.cols)
			{
				cout << " 列标号不在矩阵正常范围内 " << endl;
				continue;
			}

			if (*it == i)
				continue;

			p10[i] = p20[i];
			p11[i] = p21[i];
		}
	}
}


/*******************删除不合格点后的特征点数************************/
int Evaluate::delete_unqualified_points(const Mat& image, const Mat& homographyconst,
	string model, vector<KeyPoint> keys, vector<int>& nums, int N)
{
	int row = image.rows;						//作为不合格点的筛选条件
	int col = image.cols;
	
	int n;										//删除不合格点数之后的特征点数

	//把特征点放到矩阵中，使用单应性做统一处理
	Mat arr;
	arr.create(3, N, CV_32FC1);

	//获取矩阵每一行的首地址
	float* p20 = arr.ptr<float>(0), * p21 = arr.ptr<float>(1), * p22 = arr.ptr<float>(2);

	//把特征点放到矩阵中
	for (int i = 0; i < N; ++i)
	{
		p20[i] = keys[i].pt.x;
		p21[i] = keys[i].pt.y;
		p22[i] = 1.f;
	}

	// 1、计算待配准图中特征点到参考图得映射点，并删除不合格点
	if (model == string("perspective"))
	{
		//变换矩阵计算待配准图像特征点在参考图像中的映射点
		Mat match2_xy_change = homographyconst * arr;

		//match2_xy_change(Range(0, 2), Range::all())意思是提取 match2_xy_change 的 0、1 行，所有的列
		Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());

		//遍历映射到参考图的特征点，删除不合点
		for (int i = 0; i < N; i++)
		{
			//注意：x 对应 col，y 对应 row
			float* p0 = match2_xy_change_12.ptr<float>(0), * p1 = match2_xy_change_12.ptr<float>(1);

			if (p0[i]<0 || p0[i]>col || p1[i]<0 || p1[i]>row)
			{
				//统计不合格的列，即对应不合格的特征点
				nums.push_back(i);
			}
		}

		//删除不合格点后其他特征点个数
		n = N - nums.size();
	}
	else
	{
		cout << "模型输入错误！" << endl;
		return 0;
	}

	return n;
}


/*******************特征点检测评价************************/
float Evaluate::detect_evaluation(const Mat& image_1, const Mat& image_2, const Mat& homographyconst, 
	string model, vector<KeyPoint> keys_1, vector<KeyPoint> keys_2)
{
	vector<int> num_1, num_2;

	int row1 = image_1.rows;						//作为不合格点的筛选条件
	int col1 = image_1.cols;

	int N1 = keys_1.size();
	int N2 = keys_2.size();

	//image_1 做参考图像时，待配准图像点映射到参考图像后，删除不合格点后的特征点数
	int n1 = delete_unqualified_points(image_1, homographyconst, model, keys_2, num_1, N2);

	//image_2 做参考图像时，待配准图像点映射到参考图像后，删除不合格点后的特征点数
	int n2 = delete_unqualified_points(image_2, homographyconst, model, keys_1, num_2, N1);

	//返回较小值作为分母
	int n = min(n1, n2);

	//把特征点放到矩阵中，使用单应性做统一处理
	Mat arr_1;
	arr_1.create(3, N2, CV_32FC1);

	//获取矩阵每一行的首地址
	float* p10 = arr_1.ptr<float>(0), * p11 = arr_1.ptr<float>(1), * p12 = arr_1.ptr<float>(2);

	//把特征点放到矩阵中
	for (size_t i = 0; i < N2; ++i)
	{
		p10[i] = keys_2[i].pt.x;
		p11[i] = keys_2[i].pt.y;
		p12[i] = 1.f;
	}

	Mat match2_xy_changes, match2_xy_changes_12, final_arr;

	if (model == string("perspective"))
	{
		//变换矩阵计算待配准图像特征点在参考图像中的映射点
		match2_xy_changes = homographyconst * arr_1;

		//match2_xy_change(Range(0, 2), Range::all())意思是提取 match2_xy_change 的 0、1 行，所有的列
		match2_xy_changes_12 = match2_xy_changes(Range(0, 2), Range::all());

		//删除不合格特征点
		delete_nums_cols(match2_xy_changes_12, final_arr, num_1, n1);
	}
	else
	{
		cout << "模型输入错误！" << endl;
		return 0;
	}

	int count = 0;

	//注意：x 对应 col，y 对应 row
	float* p_0 = final_arr.ptr<float>(0), * p_1 = final_arr.ptr<float>(1);

	for (int i = 0; i < n1; i++)
	{
		for (int j = 0; j < N1; j++)
		{
			float Euc_distance = sqrt(pow((p_0[i] - keys_1[j].pt.x), 2) + pow((p_1[i] - keys_1[j].pt.y), 2));

			if (Euc_distance < threshold)
			{
				++count;
			}
		}
	}

	//特征点检测评估
	float temp = (float)count / n;

	return temp;
}


/*特征点匹配评价*/
int Evaluate::match_evaluation()
{
	return 0;
}
