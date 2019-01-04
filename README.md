## 手写数字分类

### 实验内容

使用[NMIST](http://yann.lecun.com/exdb/mnist/)当中的手写数字的数据，分别使用KNN与人工神经网络的方法实现手写数字的识别。

### 实现过程

在这个实验当中，我首先实现的是KNN算法。

#### KNN算法

KNN算法，是一种简单的机器学习的算法。他的主要思想在于，将目标的分类的概念映射到一个空间模型当中，然后，计算测试的用例与每一个训练用例的欧式距离，从而得到了测试用例与各种各样训练用例的距离，然后，选取一定范围内的训练用例，根据其中最多的来对测试用例进行分类。

> 这么说可能很抽象，引用一个在网络上的例子。比如，你一觉醒来，发现，哇，来到了一颗谜之星球。那么我现在想要知道我在哪，那么使用KNN的思想，那就是，我看天上的星星，然后找出其中离我最近的5颗，发现，有4颗是属于火星的，1颗是月亮的，那么我就可以推测出我在火星了

![knn](.\img\%5CUsers%5CBilly%5CDesktop%5Cknn.jpg)

在KNN算法当中，除了对特征进行欧式距离的计算，还有一点就是k值的选择。比如上图当中，当一个未知的绿色物体进入的时候，我们应该怎么对其分类呢？要是k比较少就会是黄色的，要是比较大就是蓝色的。过大和过小都有可能造成误差。

那么，在初步介绍了KNN算法之后，就开始进行对图片筛选的KNN算法的实现了。首先，先对已知的图片进行处理。因为我得到的是一个灰度图，因此，我要将这个灰度图转化为黑白图，从而能够直观的看出这张图片的数字的区域与非数字的区域。（当然，识别的图片也要进行相同的处理），之后，再根据黑白图划分出图片的特征值。

然后，对于一张新来的图片，这张图片与我已知的其他图片的特征值的欧式距离的长度，然后，从小到大进行排列，找出其中最小的k个，再从这k个当中找到其中包含最多的，从而将这张新的图片视为那个找到的数字。

在这里，其实可以不对图片进行划分的，若不对图片进行划分，那么图片将是28x28的大小，有784维那么将会有比较高的运算复杂度，因此，我将图片的进行了4x4的划分（在28x28的图片当中，以4x4的小图片进行划分，然后根据这4x4的小图片里面的1的内容当作该图片这个区域的特征值）。

```c++
//输入的是训练用图片，测试用图片的集合（图片已划分），以及k值，返回错误的个数
int knn(ImageList trainList, ImageList testList, int k)
{
	std::vector<KnnResult> result(trainList.lenght);
	int error = 0;
	int labelCount[10] = { 0 };
	int* pixelBuff = new int[testList.imageList[0].width * testList.imageList[0].width];

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

	}

	delete[] pixelBuff;
	return error;
}

//用于partial_sort
inline bool knnResultLess(const KnnResult &r1,const KnnResult &r2)
{
	return r1.s < r2.s;
}

//计算特征值的差
float calculateS(Image train, Image test,int *buff)
{
	float result = 0;

	for (int i = 0; i < test.width * test.height; i++)
	{
		buff[i] = test.dividePixel[i] -		 	      train.dividePixel[i];
	} 

	
	for(int i = 0;i < test.width * test.height;i++)
	{
		result += buff[i] * buff[i];
	}

	return result;
}
```



之后，是使用人工神经网络的方式。

#### 人工神经网络

人工神经网络,是一种模拟人脑神经系统的结构和功能，运用大量的简单处理单元经过广泛连接而组成的网络。而这些网络，经过人为的训练，即可得出一个我们想要的结果。（其实就是相当于，我们学习当中的背书，背的多了自然就会了）

在这个模拟人的神经系统的网络有很多种结构和实现的方式，在这里，我使用的是BP神经网络。

在BP神经网络当中，网络是由输入层，输出层和隐层组成的。每一个神经层都有着若干的神经细胞，每一个单独的细胞的功能都很小，但是，组合起来可以实现我们想要的效果了。

![BP神经网络](.\img\%5CUsers%5CBilly%5CDesktop%5Cnet.png)

每一个神经细胞都会有四个参数，活跃度，错误值，上一层神经细胞传入到这个神经细胞的计算的权值（每一个细胞传入到这个细胞的权值都不一定相同），以及上一层传入到这个神经细胞的细胞数量。

```c++
class Neuron
{
public:
	int numInputPerNerous;
	double activation = 0;
	double error = 0;

	double* weight;

	void reset();

	Neuron() = default;
	explicit Neuron(int numInputPerNerous);
	Neuron(Neuron&& n) noexcept;
	~Neuron();
};
```

之后就是在这一层当中的神经细胞的结构，在这个神经细胞层当中，存储了当前层的所有神经细胞，在这里我为了方便，也存储了当前层的神经细胞的活跃度与错误值

```c++
class NeuronLayer
{
	double * activation;
	double * error;
	
public:

	int size;
	int inputPerNeuron;
	Neuron* neurons;

	void reset() const;

	double* getActivation() const;
	double* getError() const;

	NeuronLayer(int inputPerNeuron, Neuron* neurons,int size);
	NeuronLayer(const NeuronLayer& nl);
	NeuronLayer(NeuronLayer&& nl) noexcept;
	
	~NeuronLayer();
};
```

有了这些神经结构，就可以对神经网络进行训练了。不过，首先还是要先看看原理。

在对神经网络的一次训练当中，就是让系统自身的权值进行修改，从而达到符合目标要求的目的。

在训练当中，有两次传播，第一次是正向传播，由输入的训练内容的信息，进行从输入层到输出层的正向传播，然后，就可以计算出输出层的神经元的活跃度，然后根据这个活跃度与训练内容的正确答案进行比对，计算出错误值，然后，进行第二次传播，反向传播。

在反向传播当中，根据错误值从输出层传播到输入层，根据错误值来对权值进行修改，从而减少误差率，然后反反复复的进行这样的训练，即可得到一组正确的权值，有着极大的概率能够对输入求出正确的输出。

> 在正向传播当中，其函数是 1 / （1  + exp( - x * a)） 其中x是输入（神经的activation）的值，a是一个常数，用于调整公式的斜率。
>
> 而在反向传播当中，一般是正向传播函数的导函数 f(x) = x * (1 - x)，其中x的值为输入的（神经元中的error）值，导函数接近0则说明正向传播得出的公式已经很接近正确答案了

当有着这些基本的方法的时候，就可以构建我们的神经网络了。

```c++
constexpr auto ACTIVE_RESPONSE = 0.7;

class BPNeuronNet
{
	int numInput, numHiddenLayer;
	double learingRate, errorSum = 0;
    //所有的神经网络
	std::vector<NeuronLayer> neuronLayers;

    //正向传播的公式
	inline double sigmoidActive(double activation, double response) const;
    
    //反向传播的公式
	static inline double backActive(double x);


	void updateNeuronLayer(NeuronLayer& nl, const double inputs[]) const;
	void trainNeuronLayer(NeuronLayer& nl, const double activations[],double errorArr[]) const;
	void trainUpdate(const double inputs[],const double targets[]);
public:

	double getError() const;

	void training(double inputs[], const double targets[]);
	void process(double inputs[], double * outputs[]);


	void addLayer(int num);

	BPNeuronNet(int numberInput,double learingRate);
	~BPNeuronNet();
};

```

在训练神经网络，首先第一波就是正向传播

```c++

void BPNeuronNet::training(double inputs[], const double targets[])
{
	double * perOutActivations = nullptr;
	double * perError = nullptr;
    //正向传播
	trainUpdate(inputs, targets);

    //反向传播
	for (int i = numHiddenLayer; i >= 0; i--)
	{
		NeuronLayer& curLayer = neuronLayers[i];

        //因为输入层的神经元没有错误值，因此perError = nullptr
		if (i > 0)
		{
			NeuronLayer& perLayer = neuronLayers[i - 1];

			perOutActivations = perLayer.getActivation();
			perError = perLayer.getError();

			memset(perError, 0, perLayer.size * sizeof(double));
		}
		else
		{
			perOutActivations = inputs;
			perError = nullptr;
		}


		trainNeuronLayer(curLayer, perOutActivations, perError);

	}

}

void BPNeuronNet::trainUpdate(const double inputs[], const double targets[])
{

	for(int i = 0;i <= numHiddenLayer ;i++)
	{
		updateNeuronLayer(neuronLayers[i], inputs);
		inputs = neuronLayers[i].getActivation();
	}

	NeuronLayer& outLayer = neuronLayers[numHiddenLayer];
	double * outActivation = outLayer.getActivation();
	const int numNeurons = outLayer.size;

	errorSum = 0;
	for(int i = 0;i < numNeurons;i++)
	{
		const double err = targets[i] - outActivation[i];
		outLayer.neurons[i].error = err;

		errorSum += err * err;
	}

}
```

第一步是根据输入的信息训练第一层的神经元，而从第二层开始使用的就是上一次的神经元的`activation`值了，而正向传播完，自然就会计算最后一层的错误值

在每一层的输入自然也会对神经元的`activation`修改

```c++
void BPNeuronNet::trainUpdate(const double inputs[], const double targets[])
{

	for(int i = 0;i <= numHiddenLayer ;i++)
	{
		updateNeuronLayer(neuronLayers[i], inputs);
		inputs = neuronLayers[i].getActivation();
	}

	NeuronLayer& outLayer = neuronLayers[numHiddenLayer];
	double * outActivation = outLayer.getActivation();
	const int numNeurons = outLayer.size;

	errorSum = 0;
	for(int i = 0;i < numNeurons;i++)
	{
		const double err = targets[i] - outActivation[i];
		outLayer.neurons[i].error = err;

		errorSum += err * err;
	}

}
```

之后就是反向传播了，反向传播在`training`函数当中的后半段，根据上一层的`activation`修改当前层的权重，同时也修改上一层的`error`

```c++
void BPNeuronNet::trainNeuronLayer(NeuronLayer& nl, const double activations[], double errorArr[]) const
{
	const int numNeurous = nl.size;
	const int numInputPerNeurous = nl.inputPerNeuron;
	Neuron* neuronArr = nl.neurons;

	for(int i = 0;i < numNeurous;i++)
	{
		double* curWeight = neuronArr[i].weight;
		const double error = neuronArr[i].error * backActive(neuronArr[i].activation);

		int j;
		for(j = 0;j < numInputPerNeurous;j++)
		{
            //由于第一层没有错误率
			if(errorArr)
			{
                //修改上一层的错误率
				errorArr[j] += curWeight[j] * error;		
			}
			
            //修改当前层的权重
			neuronArr[i].weight[j] += error * learingRate * activations[j];
		}
		//权值的修改与学习率有关（学习率一般为0-1的一个值）
		neuronArr[i].weight[j] += error * learingRate;
	}


}
```

这样重复多次自然就完成了训练了。

而训练完成之后就要进行识别了，由于识别无需训练（也就是无需反向传播修改权值），因此，只需要正向传播一次，然后，根据输出层的神经元的活跃度，在输出层当中最活跃的神经自然就是输出的结果了。

```c++
void BPNeuronNet::process(double inputs[], double* outputs[])
{
	for (int i = 0; i <= numHiddenLayer; i++)
	{
		updateNeuronLayer(neuronLayers[i], inputs);
		inputs = neuronLayers[i].getActivation();
	}

	*outputs = neuronLayers[numHiddenLayer].getActivation();
}
```

### 结果

首先是使用神经网络的运算结果

![BP神经网络结果](.\img\%5CUsers%5CBilly%5CDesktop%5Cnettime.png)

然后是使用KNN算法的运算结果

![KNN算法结果](.\img\knntest.png)

从这里可以看出，其实神经网络的耗时是比较少的。

虽然，两个算法一个计算了正确的数量一个计算了错误的数量，不过样本的总量都是10000，神经网络的准确度比较低应该是因为我的隐含层使用了比较少的神经元导致的。

