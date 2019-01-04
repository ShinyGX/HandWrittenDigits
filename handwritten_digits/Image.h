#pragma once
#include <cstdint>

class Image
{
	int threshold;
	uint8_t * pixel;
	int row, colomn,divideRow, divideColomn;

public:

	Image(int row,int column,int threshold = 0);
	Image(int threshold = 0);
	~Image();

	int number;
	
	int width,height;
	uint8_t * dividePixel;

	uint8_t * getPixel() const;

	void initImage(int row, int column);
	void setPixel(int x, int y, uint8_t value) const;
	void setNumber(int value);
	void print() const;
	int getRow() const;
	int getColomn() const;

	void divide();


};




