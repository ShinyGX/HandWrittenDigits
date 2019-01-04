#include "BPNeuronNet.h"


double BPNeuronNet::sigmoidActive(const double activation, const double response) const
{
	return 1.0 / (1.0 + exp(-activation * response));
}

double BPNeuronNet::backActive(const double x)
{
	return x * (1 - x);
}

void BPNeuronNet::updateNeuronLayer(NeuronLayer& nl, const double inputs[]) const
{
	const int numNerous = nl.size;
	const int numInputPerNerous = nl.inputPerNeuron;

	for(int i = 0;i < numNerous;i++)
	{
		double* curWeight = nl.neurons[i].weight;
		double netInput = 0;

		int k;
		for(k = 0;k < numInputPerNerous;k++)
		{
			netInput += curWeight[k] * inputs[k];
		}

		netInput += curWeight[k];
		nl.neurons[i].activation = sigmoidActive(netInput, ACTIVE_RESPONSE);
	}
}

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
			if(errorArr)
			{
				errorArr[j] += curWeight[j] * error;				
			}

			neuronArr[i].weight[j] += error * learingRate * activations[j];
		}

		neuronArr[i].weight[j] += error * learingRate;
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

double BPNeuronNet::getError() const
{
	return errorSum;
}

void BPNeuronNet::training(double inputs[], const double targets[])
{
	double * perOutActivations = nullptr;
	double * perError = nullptr;
	trainUpdate(inputs, targets);

	for (int i = numHiddenLayer; i >= 0; i--)
	{
		NeuronLayer& curLayer = neuronLayers[i];

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

void BPNeuronNet::process(double inputs[], double* outputs[])
{
	for (int i = 0; i <= numHiddenLayer; i++)
	{
		updateNeuronLayer(neuronLayers[i], inputs);
		inputs = neuronLayers[i].getActivation();
	}

	*outputs = neuronLayers[numHiddenLayer].getActivation();
}

void BPNeuronNet::addLayer(const int num)
{
	int inputNum = !neuronLayers.empty() ? neuronLayers[numHiddenLayer].size : numInput;

	void * raw = operator new[](num * sizeof(Neuron));
	Neuron * neurons = static_cast<Neuron*>(raw);
	for(int i = 0;i < num;i++)
	{
		new (&neurons[i])Neuron(inputNum);
	}

	neuronLayers.emplace_back(inputNum, neurons, num);
	numHiddenLayer = neuronLayers.empty() ? 0 : neuronLayers.size() - 1;
}

BPNeuronNet::BPNeuronNet(const int numberInput, const double learingRate) 
	:numInput(numberInput), numHiddenLayer(0), learingRate(learingRate)
{
}




BPNeuronNet::~BPNeuronNet()
= default;