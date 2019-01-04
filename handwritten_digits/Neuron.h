#pragma once
#include <vector>

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

