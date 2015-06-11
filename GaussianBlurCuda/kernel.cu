#include <windows.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t addWithCuda(int *b, int *g, int *r, long size);

FILE*forg = fopen("C:\\Users\\barto_000\\Dysk Google\\polibuda\\CUDA\\GaussianBlurCuda\\Gaussian-Blur-CUDA\\GaussianBlurCuda\\picasso_123.bmp", "rb");            //Uchwyt do orginalnego pliku
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
	//cout << "Typ:" << File.bfType << endl;
	fread(&File.bfSize, sizeof(File.bfSize), 1, forg);
	cout << "Rozmiar pliku: " << File.bfSize << " bajtow" << endl;

	fread(&File.bfReserved1, sizeof(File.bfReserved1), 1, forg);
	//cout << "Zarezerwowane1: " << File.bfReserved1 << endl;

	fread(&File.bfReserved2, sizeof(File.bfReserved2), 1, forg);
	//cout << "Zarezerwowane2: " << File.bfReserved2 << endl;

	fread(&File.bfOffBits, sizeof(File.bfOffBits), 1, forg);
	//cout << "Pozycja danych obrazkowych: " << File.bfOffBits << endl;

	//printf("\n");

	fseek(forg, 14, SEEK_SET);
	fread(&Picture.biSize, sizeof(Picture.biSize), 1, forg);
	//cout << "Wielkosc naglowka informacyjnego: " << Picture.biSize << endl;

	fread(&Picture.biWidth, sizeof(Picture.biWidth), 1, forg);
	cout << "Szerokosc: " << Picture.biWidth << " pikseli " << endl;

	fread(&Picture.biHeight, sizeof(Picture.biHeight), 1, forg);
	cout << "Wysokosc: " << Picture.biHeight << " pikseli " << endl;

	fread(&Picture.biPlanes, sizeof(Picture.biPlanes), 1, forg);
	//cout << "Liczba platow (zwykle 0): " << Picture.biPlanes << endl;

	fread(&Picture.biBitCount, sizeof(Picture.biBitCount), 1, forg);
	//cout << "Liczba bitow na piksel:  (1, 4, 8, or 24)" << Picture.biBitCount << endl;

	fread(&Picture.biCompression, sizeof(Picture.biCompression), 1, forg);
	//cout << "Kompresja: " << Picture.biCompression << "(0=none, 1=RLE-8, 2=RLE-4)" << endl;

	fread(&Picture.biSizeImage, sizeof(Picture.biSizeImage), 1, forg);
	//cout << "Rozmiar samego rysunku: " << Picture.biSizeImage << endl;

	fread(&Picture.biXPelsPerMeter, sizeof(Picture.biXPelsPerMeter), 1, forg);
	//cout << "Rozdzielczosc pozioma: " << Picture.biXPelsPerMeter << endl;

	fread(&Picture.biYPelsPerMeter, sizeof(Picture.biYPelsPerMeter), 1, forg);
	//cout << "Rozdzielczosc pionowa: " << Picture.biYPelsPerMeter << endl;

	fread(&Picture.biClrUsed, sizeof(Picture.biClrUsed), 1, forg);
	//cout << "Liczba kolorow w palecie: " << Picture.biClrUsed << endl;

	fread(&Picture.biClrImportant, sizeof(Picture.biClrImportant), 1, forg);
	//cout << "Wazne kolory w palecie: " << Picture.biClrImportant << endl;
}

char z;

__global__ void ReadImage(int *B, int *G, int *R, int bfSize)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < bfSize)
	{
		B[index] = 255;// B[index] + 50;
		G[index] = 255;// G[index] + 50;
		R[index] = 255;// R[index] + 50;
	}

}

int main()
{

	//ReadBMP("picasso_123.bmp");

	char* filename = "picasso_123.bmp";

	int i;
	FILE* f = fopen(filename, "rb");

	if (f == NULL)
		throw "Argument Exception";

	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = *(int*)&info[18];
	int height = *(int*)&info[22];

	cout << endl;
	cout << "  Name: " << filename << endl;
	cout << " Width: " << width << endl;
	cout << "Height: " << height << endl;

	int row_padded = (width * 3 + 3) & (~3);
	unsigned char* data = new unsigned char[row_padded];
	unsigned char tmp;

	// deklaracja zmiennych
	int *B, *G, *R;
	long liczba_pikseli = width*height;

	B = new int[liczba_pikseli*sizeof(int)];
	G = new int[liczba_pikseli*sizeof(int)];
	R = new int[liczba_pikseli*sizeof(int)];

	long licznik_pikseli = 0;
	for (int i = 0; i < height; i++)
	{
		fread(data, sizeof(unsigned char), row_padded, f);
		for (int j = 0; j < width * 3; j += 3)
		{
			// Convert (B, G, R) to (R, G, B)
			tmp = data[j]; 
			data[j] = data[j + 2]; 
			data[j + 2] = tmp; 

			R[licznik_pikseli] = (int)data[j];
			G[licznik_pikseli] = (int)data[j + 1];
			B[licznik_pikseli] = (int)data[j + 2];

			cout << licznik_pikseli << ": " << "R: " << R[licznik_pikseli] << " G: " << G[licznik_pikseli] << " B: " << B[licznik_pikseli] << endl;
			licznik_pikseli++;
		}
	}

	fclose(f);

	addWithCuda(B, G, R, liczba_pikseli);

	cout << "--------------"<<endl;

	for (int i = 0; i < liczba_pikseli; i++){
		cout << i << ": " << "R: " << R[i] << " G: " << G[i] << " B: " << B[i] << endl;
	}


	for (int i = 0; i<File.bfOffBits; i++)
	{
		z = fgetc(forg);
		fprintf(fsz, "%c", z);                   //Utworzenie naglowka nowej Bitmapy
	}
	fseek(fsz, 54, SEEK_SET);
	for (int i = 0; i < liczba_pikseli; i++)
	{
		fprintf(fsz, "%c", (int)(B[i]));
		fprintf(fsz, "%c", (int)(G[i]));
		fprintf(fsz, "%c", (int)(R[i]));
	}

	delete[] B;
	delete[] G;
	delete[] R;

	//system("PAUSE");
}

cudaError_t addWithCuda(int *b, int *g, int *r, long size)
{
	int *d_B, *d_G, *d_R;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_B, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_G, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_R, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_B, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_G, g, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_R, r, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	ReadImage << <1, size >> >(d_B, d_G, d_R, size);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(b, d_B, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(g, d_G, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(r, d_R, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_B);
	cudaFree(d_G);
	cudaFree(d_R);

	return cudaStatus;
}