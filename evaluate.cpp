# include"evaluate.h"
# include<vector>

#include<opencv2\core\core.hpp>
#include<opencv2\features2d\features2d.hpp>

#include<vector>
#include<string>
#include<iostream>

using namespace std;
using namespace cv;


/*******************ɾ��������ָ������************************/
void Evaluate::delete_col(Mat& object, int num)
{
	if (num < 0 || num >= object.cols)
	{
		cout << " �б�Ų��ھ���������Χ�� " << endl;
	}
	else
	{
		//ɾ�����Ǿ�������һ��
		if (num == object.cols - 1)
		{
			object = object.t();			 //�������
			object.pop_back();				 //�������һ��Ԫ��
			object = object.t();
		}
		else
		{
			//num ��֮�����������ǰ��һ��
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


/*******************ɾ�������ж��ָ����************************/
/*object ��Ҫ���в����ľ���
 *arrs ɾ�����ϸ��������ľ���*/
void Evaluate::delete_nums_cols(Mat& object, Mat& arrs, vector<int>& nums, int n)
{
	arrs.create(2, n, CV_32FC1);

	//��ȡ����ÿһ�е��׵�ַ
	float* p10 = arrs.ptr<float>(0), * p11 = arrs.ptr<float>(1);
	float* p20 = object.ptr<float>(0), * p21 = object.ptr<float>(1);

	for (int i = 0; i < object.cols; i++)
	{
		auto it = nums.begin();
		for (; it != nums.end(); it++)
		{
			if ((*it) < 0 || (*it) >= object.cols)
			{
				cout << " �б�Ų��ھ���������Χ�� " << endl;
				continue;
			}

			if (*it == i)
				continue;

			p10[i] = p20[i];
			p11[i] = p21[i];
		}
	}
}


/*******************ɾ�����ϸ������������************************/
int Evaluate::delete_unqualified_points(const Mat& image, const Mat& homographyconst,
	string model, vector<KeyPoint> keys, vector<int>& nums, int N)
{
	int row = image.rows;						//��Ϊ���ϸ���ɸѡ����
	int col = image.cols;
	
	int n;										//ɾ�����ϸ����֮�����������

	//��������ŵ������У�ʹ�õ�Ӧ����ͳһ����
	Mat arr;
	arr.create(3, N, CV_32FC1);

	//��ȡ����ÿһ�е��׵�ַ
	float* p20 = arr.ptr<float>(0), * p21 = arr.ptr<float>(1), * p22 = arr.ptr<float>(2);

	//��������ŵ�������
	for (int i = 0; i < N; ++i)
	{
		p20[i] = keys[i].pt.x;
		p21[i] = keys[i].pt.y;
		p22[i] = 1.f;
	}

	// 1���������׼ͼ�������㵽�ο�ͼ��ӳ��㣬��ɾ�����ϸ��
	if (model == string("perspective"))
	{
		//�任����������׼ͼ���������ڲο�ͼ���е�ӳ���
		Mat match2_xy_change = homographyconst * arr;

		//match2_xy_change(Range(0, 2), Range::all())��˼����ȡ match2_xy_change �� 0��1 �У����е���
		Mat match2_xy_change_12 = match2_xy_change(Range(0, 2), Range::all());

		//����ӳ�䵽�ο�ͼ�������㣬ɾ�����ϵ�
		for (int i = 0; i < N; i++)
		{
			//ע�⣺x ��Ӧ col��y ��Ӧ row
			float* p0 = match2_xy_change_12.ptr<float>(0), * p1 = match2_xy_change_12.ptr<float>(1);

			if (p0[i]<0 || p0[i]>col || p1[i]<0 || p1[i]>row)
			{
				//ͳ�Ʋ��ϸ���У�����Ӧ���ϸ��������
				nums.push_back(i);
			}
		}

		//ɾ�����ϸ����������������
		n = N - nums.size();
	}
	else
	{
		cout << "ģ���������" << endl;
		return 0;
	}

	return n;
}


/*******************������������************************/
float Evaluate::detect_evaluation(const Mat& image_1, const Mat& image_2, const Mat& homographyconst, 
	string model, vector<KeyPoint> keys_1, vector<KeyPoint> keys_2)
{
	vector<int> num_1, num_2;

	int row1 = image_1.rows;						//��Ϊ���ϸ���ɸѡ����
	int col1 = image_1.cols;

	int N1 = keys_1.size();
	int N2 = keys_2.size();

	//image_1 ���ο�ͼ��ʱ������׼ͼ���ӳ�䵽�ο�ͼ���ɾ�����ϸ������������
	int n1 = delete_unqualified_points(image_1, homographyconst, model, keys_2, num_1, N2);

	//image_2 ���ο�ͼ��ʱ������׼ͼ���ӳ�䵽�ο�ͼ���ɾ�����ϸ������������
	int n2 = delete_unqualified_points(image_2, homographyconst, model, keys_1, num_2, N1);

	//���ؽ�Сֵ��Ϊ��ĸ
	int n = min(n1, n2);

	//��������ŵ������У�ʹ�õ�Ӧ����ͳһ����
	Mat arr_1;
	arr_1.create(3, N2, CV_32FC1);

	//��ȡ����ÿһ�е��׵�ַ
	float* p10 = arr_1.ptr<float>(0), * p11 = arr_1.ptr<float>(1), * p12 = arr_1.ptr<float>(2);

	//��������ŵ�������
	for (size_t i = 0; i < N2; ++i)
	{
		p10[i] = keys_2[i].pt.x;
		p11[i] = keys_2[i].pt.y;
		p12[i] = 1.f;
	}

	Mat match2_xy_changes, match2_xy_changes_12, final_arr;

	if (model == string("perspective"))
	{
		//�任����������׼ͼ���������ڲο�ͼ���е�ӳ���
		match2_xy_changes = homographyconst * arr_1;

		//match2_xy_change(Range(0, 2), Range::all())��˼����ȡ match2_xy_change �� 0��1 �У����е���
		match2_xy_changes_12 = match2_xy_changes(Range(0, 2), Range::all());

		//ɾ�����ϸ�������
		delete_nums_cols(match2_xy_changes_12, final_arr, num_1, n1);
	}
	else
	{
		cout << "ģ���������" << endl;
		return 0;
	}

	int count = 0;

	//ע�⣺x ��Ӧ col��y ��Ӧ row
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

	//������������
	float temp = (float)count / n;

	return temp;
}


/*������ƥ������*/
int Evaluate::match_evaluation()
{
	return 0;
}
