#pragma once
#include <queue>
#include <thread>
#include <string>
#include <mutex>

class ProgressBar
{
	int max;
	int curProgress = 0;
	int currentValue = 0;

	std::queue<int> upgradeQueue;
	bool stopThread = false;

	static void* progressThread(void *__this);
	std::thread * printThread{};
	std::mutex mtx;

	void updateValue(int value);
public:
	ProgressBar(int max, const std::string& label);
	explicit ProgressBar(int max);
	~ProgressBar();

	ProgressBar& operator++();
	ProgressBar& operator+=(int value);

	
	static void show();
	void shutDown();

	void reset(int max,const std::string& label);
};


