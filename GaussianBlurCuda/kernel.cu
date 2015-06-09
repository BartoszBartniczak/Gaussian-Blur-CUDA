#include <windows.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define THREADS_PER_BLOCK 3
FILE*forg = fopen("C:\\Users\\barto_000\\Dysk Google\\polibuda\\CUDA\\GaussianBlurCuda\\Gaussian-Blur-CUDA\\GaussianBlurCuda\\picasso.bmp", "rb");            //Uchwyt do orginalnego pliku
FILE*fsz = fopen("C:\\Users\\barto_000\\Dysk Google\\polibuda\\CUDA\\GaussianBlurCuda\\Gaussian-Blur-CUDA\\GaussianBlurCuda\\output1.bmp", "wb");                    //Uchwyt do nowego pliku
struct FileHeader {
	short bfType;
	int bfSize;
	short bfReserved1;
	short bfReserved2;
	short bfOffBits;
};
FileHeader File;

struct PictureHeader {
	int biSize;
	int biWidth;
	int biHeight;
	short biPlanes;
	short biBitCount;
	int biCompression;
	int biSizeImage;
	int biXPelsPerMeter;
	int biYPelsPerMeter;
	int biClrUsed;
	int biClrImportant;
};
PictureHeader Picture;

void header(){

	fread(&File.bfType, sizeof(File.bfType), 1, forg);
	cout << "Typ:" << File.bfType << endl;
	fread(&File.bfSize, sizeof(File.bfSize), 1, forg);
	cout << "Rozmiar pliku: " << File.bfSize << " bajtow" << endl;

	fread(&File.bfReserved1, sizeof(File.bfReserved1), 1, forg);
	cout << "Zarezerwowane1: " << File.bfReserved1 << endl;

	fread(&File.bfReserved2, sizeof(File.bfReserved2), 1, forg);
	cout << "Zarezerwowane2: " << File.bfReserved2 << endl;

	fread(&File.bfOffBits, sizeof(File.bfOffBits), 1, forg);
	cout << "Pozycja danych obrazkowych: " << File.bfOffBits << endl;

	printf("\n");

	fseek(forg, 14, SEEK_SET);
	fread(&Picture.biSize, sizeof(Picture.biSize), 1, forg);
	cout << "Wielkosc naglowka informacyjnego: " << Picture.biSize << endl;

	fread(&Picture.biWidth, sizeof(Picture.biWidth), 1, forg);
	cout << "Szerokosc: " << Picture.biWidth << " pikseli " << endl;

	fread(&Picture.biHeight, sizeof(Picture.biHeight), 1, forg);
	cout << "Wysokosc: " << Picture.biHeight << " pikseli " << endl;

	fread(&Picture.biPlanes, sizeof(Picture.biPlanes), 1, forg);
	cout << "Liczba platow (zwykle 0): " << Picture.biPlanes << endl;

	fread(&Picture.biBitCount, sizeof(Picture.biBitCount), 1, forg);
	cout << "Liczba bitow na piksel:  (1, 4, 8, or 24)" << Picture.biBitCount << endl;

	fread(&Picture.biCompression, sizeof(Picture.biCompression), 1, forg);
	cout << "Kompresja: " << Picture.biCompression << "(0=none, 1=RLE-8, 2=RLE-4)" << endl;

	fread(&Picture.biSizeImage, sizeof(Picture.biSizeImage), 1, forg);
	cout << "Rozmiar samego rysunku: " << Picture.biSizeImage << endl;

	fread(&Picture.biXPelsPerMeter, sizeof(Picture.biXPelsPerMeter), 1, forg);
	cout << "Rozdzielczosc pozioma: " << Picture.biXPelsPerMeter << endl;

	fread(&Picture.biYPelsPerMeter, sizeof(Picture.biYPelsPerMeter), 1, forg);
	cout << "Rozdzielczosc pionowa: " << Picture.biYPelsPerMeter << endl;

	fread(&Picture.biClrUsed, sizeof(Picture.biClrUsed), 1, forg);
	cout << "Liczba kolorow w palecie: " << Picture.biClrUsed << endl;

	fread(&Picture.biClrImportant, sizeof(Picture.biClrImportant), 1, forg);
	cout << "Wazne kolory w palecie: " << Picture.biClrImportant << endl;
}

char z;

__global__ void ReadImage(int *B, int *G, int *R, int bfSize)
{
	int index = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;
	if (index < bfSize)
	{
		B[index] = R[index];
		G[index] = B[index];
		R[index] = G[index];
	}

}
int main()
{
	
	header();
	time_t  czas, czas1, czas2, czas3;
	time_t  start, start1, start2, start3;
	int ile = 1;
	long liczba_blokow = Picture.biWidth * Picture.biHeight;
	int *B, *G, *R;
	int *d_B, *d_G, *d_R;
	B = new int[File.bfSize*sizeof(int)];
	G = new int[File.bfSize*sizeof(int)];
	R = new int[File.bfSize*sizeof(int)];

	cudaMalloc(&d_B, File.bfSize*sizeof(int));
	cudaMalloc(&d_G, File.bfSize*sizeof(int));
	cudaMalloc(&d_R, File.bfSize*sizeof(int));

	fseek(forg, 0, SEEK_SET);
	for (int i = 0; i<File.bfOffBits; i++)
	{
		z = fgetc(forg);
		fprintf(fsz, "%c", z);                   //Utworzenie naglowka nowej Bitmapy
	}
	for (int i = File.bfOffBits; i < File.bfSize; i++)
	{
		B[i] = fgetc(forg);
		G[i] = fgetc(forg);
		R[i] = fgetc(forg);
	}
	start = clock();
	start1 = clock();

	for (unsigned int i = 0; i < ile; i++)
	{
		cudaMemcpy(d_B, B, File.bfSize*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_G, G, File.bfSize*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_R, R, File.bfSize*sizeof(int), cudaMemcpyHostToDevice);
	}

	czas1 = (float(clock() - start1)*CLOCKS_PER_SEC) / 1000;
	cout << "Przesylanie do CUDA " << dec << czas1 << " milisekund " << endl;

	start2 = clock();

	for (unsigned int i = 0; i < ile; i++)
		ReadImage << < liczba_blokow, THREADS_PER_BLOCK >> >(d_B, d_G, d_R, File.bfSize);

	czas2 = (float(clock() - start2)*CLOCKS_PER_SEC) / 1000;
	cout << "Wykonanie funkcji " << dec << czas2 << " milisekund " << endl;
	start3 = clock();
	for (unsigned int i = 0; i < ile; i++)
	{
		cudaMemcpy(B, d_B, File.bfSize*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(G, d_G, File.bfSize*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(R, d_R, File.bfSize*sizeof(int), cudaMemcpyDeviceToHost);
	}
	czas3 = (float(clock() - start3)*CLOCKS_PER_SEC) / 1000;
	cout << "Powrót z CUDA " << dec << czas3 << " milisekund " << endl;

	czas = (float(clock() - start)*CLOCKS_PER_SEC) / 1000;
	cout << "Program wykonywal sie " << dec << czas << " milisekund " << endl;
	fseek(fsz, 54, SEEK_SET);
	for (int i = File.bfOffBits; i < File.bfSize; i++)
	{
		fprintf(fsz, "%c", (int)(R[i]));
		fprintf(fsz, "%c", (int)(B[i]));
		fprintf(fsz, "%c", (int)(G[i]));
	}

	delete[] B;
	delete[] G;
	delete[] R;
	cudaFree(d_B); cudaFree(d_G); cudaFree(d_R);
	system("PAUSE");
}