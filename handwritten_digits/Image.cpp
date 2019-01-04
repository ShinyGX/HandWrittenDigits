#include "Image.h"
#include <iostream>
#include <bitset>


// Image::Image(Image&& i) noexcept
// 	:threshold(i.threshold),pixel(i.pixel),row(i.row),colomn(i.colomn),
// 	divideRow(i.divideRow),divideColomn(i.divideColomn),number(i.number),
// 	width(i.width),height(i.height),dividePixel(i.dividePixel)
// {
// }

Image::Image(int row, int column, int threshold) 
:threshold(threshold), divideRow(4), divideColomn(4)
{
	initImage(row, column);
}

Image::Image(int threshold)
	: threshold(threshold), row(0), colomn(0), divideRow(4), divideColomn(4)
{
}

Image::~Image()
{
	
}

uint8_t* Image::getPixel() const
{
	return pixel;
}

void Image::initImage(int row, int column)
{

	this->row = row;
	this->colomn = column;

	pixel = new uint8_t [row * column];

}

void Image::setPixel(int x, int y, uint8_t value) const
{
	if (value > threshold)
		value = 1;
	pixel[x * colomn + y] = value;
}

void Image::setNumber(int value)
{
	this->number = value;
}

void Image::print() const
{
	using namespace std;

	cout << endl << number << endl;

	for(int i = 0;i < row;i++)
	{
		for(int j = 0;j < colomn;j++)
		{
			cout << (pixel[i * colomn + j] == 0 ? " " : "*");
		}

		cout << endl;
	}
}

void Image::divide()
{
	width = row / divideRow;
	height = colomn / divideColomn;

	dividePixel = new uint8_t[width * height];

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uint8_t value = 0;
			for(int k = i * divideRow;k < (i + 1) * divideRow;k++)
			{
				for(int l = j * divideColomn;l < (j + 1) * divideColomn;l++)
				{
					value += pixel[k * colomn + l];
				}
			}
			dividePixel[i * width + j] = value;
		}
	}

}



int Image::getColomn() const
{
	return colomn;
}

int Image::getRow() const
{
	return row;
}

