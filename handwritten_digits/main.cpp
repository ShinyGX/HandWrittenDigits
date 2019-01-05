#include <fstream>
#include <iostream>
#include "Image.h"
#include <vector>
#include <algorithm>
#include "BPNeuronNet.h"
#include "NeuronUtil.h"
#include "ProgressBar.h"
#include <future>
#include <sys/timeb.h>

//#define NEURON_ENABLE 

#define big_to_little32(A) ((((uint32_t)(A) & 0xff000000) >> 24) | (((uint32_t)(A) & 0x00ff0000) >> 8) | \
             (((uint32_t)(A) & 0x0000ff00) << 8) | (((uint32_t)(A) & 0x000000ff) << 24))

#define load_from_big_to_little32(istream,value)\
	istream.read(reinterpret_cast<char*>(&value), sizeof value);\
	value = big_to_little32(value)



typedef struct List
{
	Image * imageList;
	int lenght;
}ImageList;
struct KnnResult
{
	int label;
	float s;

	KnnResult(): label(0), s(0)
	{
	}

	KnnResult(const int label, const float s):label(label),s(s)
	{
	}

	bool operator < (const KnnResult& r) const
	{
		return s < r.s;	
	}

};

static ProgressBar progressBar(70000, "loading test and train image...");


long long getCurrentTime();

ImageList loadImageAndLabelFile(const std::string& imagePath, const std::string& labelPath);
void loadImageAndLabelFileInThread(std::promise<ImageList> &promiseObj,const std::string& imagePath, const std::string& labelPath);

int knn(ImageList trainList, ImageList testList, int k);
void knnInThread(std::promise<int>& errorRate, ImageList trainList, ImageList testList, int k);
bool knnResultLess(const KnnResult &r1, const KnnResult &r2);
float calculateS(Image train, Image test,int *buff);

void perProcessInputDataWithNoise(const uint8_t *pixel, double * out, int size);
void perProcessInputData(const uint8_t *pixel, double* out, int size);

double train(ImageList trainList, BPNeuronNet& bpNeuronNet);
int test(ImageList testList, BPNeuronNet& bpNeuronNet);

int main()
{
	using namespace  std;

	promise<ImageList> trainingObj;
	future<ImageList> trainFutureObj = trainingObj.get_future();

	promise<ImageList> testObj;
	future<ImageList> testFutureObj = testObj.get_future();

	thread t1(loadImageAndLabelFileInThread, ref(trainingObj), "train-images.idx", "train-labels.idx");
	thread t2(loadImageAndLabelFileInThread, ref(testObj), "t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

	ImageList testImageList = testFutureObj.get();
	ImageList trainingImageList = trainFutureObj.get();

	t1.join();
	t2.join();

	testImageList.imageList[rand() % testImageList.lenght].print();

	int chose;
	cout << "1-> bp neuron net" << endl;
	cout << "2-> knn" << endl;
	cout << "请输入要使用的算法（请输入相应的正确的数字）" << endl;
	cin >> chose;

	if (chose == 1) {
		auto startTime = getCurrentTime();

		BPNeuronNet bpNeuronNet(trainingImageList.imageList[0].getRow() * trainingImageList.imageList[0].getColomn(), 0.4);
		bpNeuronNet.addLayer(100);
		bpNeuronNet.addLayer(10);
		auto r = train(trainingImageList, bpNeuronNet);
		cout << "error rate : " << r << endl;

		auto success = test(testImageList, bpNeuronNet);
		cout << "success : " << success << " count : " << testImageList.lenght << endl;
		cout << "time : " << getCurrentTime() - startTime << endl;

	}
	else if (chose == 2) {
		auto startTime = getCurrentTime();
		cout << "divide train image" << endl;
		for (int i = 0; i < trainingImageList.lenght; i++)
		{
			trainingImageList.imageList[i].divide();
		}

		cout << "divide test image" << endl;
		for (int i = 0; i < testImageList.lenght; i++)
		{
			testImageList.imageList[i].divide();
		}

		ImageList testList1, testList2, testList3, testList4;

		testList4.lenght = testImageList.lenght / 4;
		testList3.lenght = testImageList.lenght / 4;
		testList2.lenght = testImageList.lenght / 4;
		testList1.lenght = testImageList.lenght / 4;

		testList4.imageList = &testImageList.imageList[7500];
		testList3.imageList = &testImageList.imageList[5000];
		testList2.imageList = &testImageList.imageList[2500];
		testList1.imageList = &testImageList.imageList[0];

		promise<int> error1;
		future<int> ef1 = error1.get_future();
		promise<int> error2;
		future<int> ef2 = error2.get_future();
		promise<int> error3;
		future<int> ef3 = error3.get_future();
		promise<int> error4;
		future<int> ef4 = error4.get_future();


		progressBar.reset(testImageList.lenght, "knn test");
		thread knn1(knnInThread, ref(error1), trainingImageList, testList1, 20);
		thread knn2(knnInThread, ref(error2), trainingImageList, testList2, 20);
		thread knn3(knnInThread, ref(error3), trainingImageList, testList3, 20);
		thread knn4(knnInThread, ref(error4), trainingImageList, testList4, 20);

		knn1.join();
		knn2.join();
		knn3.join();
		knn4.join();

		//	 float result = knn(trainingImageList, testImageList, 20);
		float result = ef1.get() + ef2.get() + ef3.get() + ef4.get();
		std::cout << endl << "error :" << result << std::endl;
		std::cout << "error rate : " << static_cast<float>(result) / testImageList.lenght << endl;
		cout << "time : " << getCurrentTime() - startTime << endl;
	}

	progressBar.shutDown();

	int a;
	cout << "输入任意键结束" << endl;
	std::cin >> a;
	return 0;
}

long long getCurrentTime()
{
	timeb t;
	ftime(&t);
	return t.time * 1000 + t.millitm;
}

ImageList loadImageAndLabelFile(const std::string& imagePath, const std::string& labelPath)
{
	using namespace std;

	ImageList list = {};

	ifstream digiteImage;
	ifstream digiteLabel;

	digiteLabel.open(labelPath, ios::binary);
	digiteImage.open(imagePath, ios::binary);
	int32_t imageMagicNumber, imageCount, rows, columns;
	int32_t labelMagicNumber, labelCount;
	if (digiteImage.is_open() && digiteLabel.is_open())
	{
	
		load_from_big_to_little32(digiteLabel, labelMagicNumber);
		load_from_big_to_little32(digiteLabel, labelCount);

		load_from_big_to_little32(digiteImage, imageMagicNumber);
		load_from_big_to_little32(digiteImage, imageCount);
		load_from_big_to_little32(digiteImage, rows);
		load_from_big_to_little32(digiteImage, columns);


		if (imageCount != labelCount)
		{
			digiteLabel.close();
			digiteImage.close();
			throw exception("input file have some error");
		}


		Image * imageList = new Image[imageCount];
		for (int i = 0; i < imageCount; i++)
		{
			imageList[i].initImage(rows, columns);
		}

		for (int i = 0; i < imageCount; i++)
		{
			for (int x = 0; x < rows; x++)
			{
				for (int y = 0; y < columns; y++)
				{
					uint8_t grayScale;
					digiteImage.read(reinterpret_cast<char *>(&grayScale), sizeof grayScale);

					imageList[i].setPixel(x, y, grayScale);
				}
			}

			uint8_t label;
			digiteLabel.read(reinterpret_cast<char *>(&label), sizeof label);
			imageList[i].setNumber(label);


			++progressBar;


		}

		list.imageList = imageList;
		list.lenght = imageCount;
	}


	digiteLabel.close();
	digiteImage.close();


	return list;
}

void loadImageAndLabelFileInThread(std::promise<ImageList> &promiseObj, const std::string& imagePath, const std::string& labelPath)
{
	promiseObj.set_value(loadImageAndLabelFile(imagePath, labelPath));
}

int knn(ImageList trainList, ImageList testList, int k)
{
	std::vector<KnnResult> result(trainList.lenght);
	int error = 0;
	int labelCount[10] = { 0 };
	int* pixelBuff = new int[testList.imageList[0].width * testList.imageList[0].width];
	//progressBar.reset(testList.lenght, "knn test");
	for (int i = 0; i < testList.lenght; i++)
	{
		result.clear();
		for (int& j : labelCount)
		{
			j = 0;
		}

		int s = 0;
		for (int j = 0; j < trainList.lenght; j++)
		{
			s = calculateS(trainList.imageList[j], testList.imageList[i], pixelBuff);

			result.emplace_back(trainList.imageList[j].number, s);
		}

		std::partial_sort(result.begin(), result.begin() + k, result.end(), knnResultLess);
		//std::sort(result.begin(), result.end(), knnResultLess);

		for (int l = 0; l < k; l++)
		{
			labelCount[result[l].label]++;
		}

		int maxNumber = 0, resultNumber = 0;
		for (int l = 0; l < 10; l++)
		{
			if (labelCount[l] > maxNumber)
			{
				maxNumber = labelCount[l];
				resultNumber = l;
			}
		}

		if (resultNumber!= testList.imageList[i].number)
			error++;

		++progressBar;
		//std::cout << static_cast<double>(i) * 100 / testList.lenght << "%" << std::endl;
	}

	delete[] pixelBuff;
	return error;
}

inline bool knnResultLess(const KnnResult &r1,const KnnResult &r2)
{
	return r1.s < r2.s;
}

float calculateS(Image train, Image test,int *buff)
{
	float result = 0;

	for (int i = 0; i < test.width * test.height; i++)
	{
		buff[i] = test.dividePixel[i] - train.dividePixel[i];
	}

	
	for(int i = 0;i < test.width * test.height;i++)
	{
		result += buff[i] * buff[i];
	}

	return result;
}

void knnInThread(std::promise<int>& errorRate,ImageList trainList, ImageList testList, int k)
{
	errorRate.set_value(knn(trainList, testList, k));
}


double train(const ImageList trainList,BPNeuronNet& bpNeuronNet)
{
	double netTarget[10];
	double netTrain[28 * 28];
	progressBar.reset(trainList.lenght,"start training... ");
	for(int i = 0;i < trainList.lenght;i++)
	{
		memset(netTarget, 0, 10 * sizeof(double));

		netTarget[trainList.imageList[i].number] = 1.0;
		perProcessInputDataWithNoise(trainList.imageList[i].getPixel(), netTrain, 28 * 28);
		bpNeuronNet.training(netTrain, netTarget);

		++progressBar;
	}

	return bpNeuronNet.getError();
}

int test(ImageList testList,BPNeuronNet& bpNeuronNet)
{
	int success = 0;
	double * netOut = nullptr;
	double netTest[28 * 28];
	progressBar.reset(testList.lenght,"testing neuron net");
	for(int i = 0;i < testList.lenght;i++)
	{
		perProcessInputData(testList.imageList[i].getPixel(), netTest, 28 * 28);
		bpNeuronNet.process(netTest, &netOut);

		int id = -1;
		double maxValue = -999;
		for(int j = 0;j < 10;j++)
		{
			if(netOut[j] > maxValue)
			{
				maxValue = netOut[j];
				id = j;
			}
		}

		if(id == testList.imageList[i].number)
		{
			success++;
		}

		++progressBar;
	}

	return success;
}

inline void perProcessInputDataWithNoise(const uint8_t *pixel,double * out,int size)
{
	for(int i = 0;i < size;i++)
	{
		out[i] = ((pixel[i] > 0) ? 1.0 : 0.0) + randomFloat() * 0.1f;
	}
}

inline void perProcessInputData(const uint8_t *pixel,double* out,int size)
{
	for (int i = 0; i < size; i++)
	{
		out[i] = ((pixel[i] > 0) ? 1.0 : 0.0);
	}
}