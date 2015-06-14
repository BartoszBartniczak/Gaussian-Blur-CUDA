#include <windows.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t GaussianBlurWithCuda(int *b, int *g, int *r, long size, int width);

FILE*forg = fopen("C:\\Users\\barto_000\\Dysk Google\\polibuda\\CUDA\\GaussianBlurCuda\\Gaussian-Blur-CUDA\\GaussianBlurCuda\\picasso_789.bmp", "rb");            //Uchwyt do orginalnego pliku
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

__global__ void GaussianBlur(int *B, int *G, int *R, int numberOfPixels, int width, int *B_new, int *G_new, int *R_new)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= numberOfPixels){
		//printf("%d\n",index);
		return;
	}

	int mask[] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	int s = mask[0] + mask[1] + mask[2] + mask[3] + mask[4] + mask[5] + mask[6] + mask[7] + mask[8];

	if (index < width){ // dolny rzad pikseli
		if (index == 0){ //lewy dolny rog
			s = mask[4] + mask[1] + mask[2] + mask[5];
			B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5]) / s);
			G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5]) / s);
			R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5]) / s);
			return;
		}

		if (index == width - 1){//prawy dolny rog
			s = mask[4] + mask[0] + mask[1] + mask[3];
			B_new[index] = (B[index] * mask[4] + B[index + width - 1] * mask[0] + B[index + width] * mask[1] + B[index - 1] * mask[3]);
			G_new[index] = (G[index] * mask[4] + G[index + width - 1] * mask[0] + G[index + width] * mask[1] + G[index - 1] * mask[3]);
			R_new[index] = (R[index] * mask[4] + R[index + width - 1] * mask[0] + R[index + width] * mask[1] + R[index - 1] * mask[3]);
			return;
		}
		//reszta pikseli w dolnym rzedzie
		s = mask[4] + mask[1] + mask[2] + mask[5] + mask[0] + mask[3];
		B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5] + B[index + width - 1] * mask[0] + B[index - 1] * mask[3]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5] + R[index + width - 1] * mask[0] + R[index - 1] * mask[3]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5] + G[index + width - 1] * mask[0] + G[index - 1] * mask[3]) / s);

		return;
	}
	if (index >= numberOfPixels - width){ //gorny rzad pikseli

		if (index == numberOfPixels - width){ //lewy gorny rog
			s = mask[4] + mask[5] + mask[7] + mask[8];
			B_new[index] = (int)((B[index] * mask[4] + B[index + 1] * mask[5] + B[index - width] * mask[7] + B[index - width + 1] * mask[8]) / s);
			G_new[index] = (int)((G[index] * mask[4] + G[index + 1] * mask[5] + G[index - width] * mask[7] + G[index - width + 1] * mask[8]) / s);
			R_new[index] = (int)((R[index] * mask[4] + R[index + 1] * mask[5] + R[index - width] * mask[7] + R[index - width + 1] * mask[8]) / s);
			return;
		}

		if (index == numberOfPixels - 1){ //prawy gorny rog
			s = mask[4] + mask[3] + mask[6] + mask[7];
			B_new[index] = (int)((B[index] * mask[4] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7]) / s);
			G_new[index] = (int)((G[index] * mask[4] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7]) / s);
			R_new[index] = (int)((R[index] * mask[4] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7]) / s);
			return;
		}

		s = mask[4] + mask[3] + mask[5] + mask[6] + mask[7] + mask[8];
		B_new[index] = (int)((B[index] * mask[4] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7] + B[index + 1] * mask[5] + B[index - width] * mask[8]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7] + R[index + 1] * mask[5] + R[index - width] * mask[8]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7] + G[index + 1] * mask[5] + G[index - width] * mask[8]) / s);
		return;
	}
	if (index % width == 0){ //lewa sciana
		s = mask[4] + mask[1] + mask[2] + mask[5] + mask[8] + mask[7];
		B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width + 1] * mask[2] + B[index + 1] * mask[5] + B[index - width + 1] * mask[8] + B[index - width]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width + 1] * mask[2] + G[index + 1] * mask[5] + G[index - width + 1] * mask[8] + G[index - width]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width + 1] * mask[2] + R[index + 1] * mask[5] + R[index - width + 1] * mask[8] + R[index - width]) / s);
		return;
	}
	if (index % width == width - 1){ //prawa sciana
		s = mask[4] + mask[1] + mask[0] + mask[3] + mask[6] + mask[7];
		B_new[index] = (int)((B[index] * mask[4] + B[index + width] * mask[1] + B[index + width - 1] * mask[0] + B[index - 1] * mask[3] + B[index - width - 1] * mask[6] + B[index - width] * mask[7]) / s);
		R_new[index] = (int)((R[index] * mask[4] + R[index + width] * mask[1] + R[index + width - 1] * mask[0] + R[index - 1] * mask[3] + R[index - width - 1] * mask[6] + R[index - width] * mask[7]) / s);
		G_new[index] = (int)((G[index] * mask[4] + G[index + width] * mask[1] + G[index + width - 1] * mask[0] + G[index - 1] * mask[3] + G[index - width - 1] * mask[6] + G[index - width] * mask[7]) / s);
		return;
	}


		int poz_1 = index - width - 1;
		int poz_2 = index - width;
		int poz_3 = index - width + 1;
		int poz_4 = index - 1;
		int poz_5 = index;
		int poz_6 = index + 1;
		int poz_7 = index + width - 1;
		int poz_8 = index + width;
		int poz_9 = index + width + 1;

		B_new[index] = (int)(((B[poz_1] * mask[0]) + (B[poz_2] * mask[1]) + (B[poz_3] * mask[2]) + (B[poz_4] * mask[3]) + (B[poz_5] * mask[4]) + (B[poz_6] * mask[5]) + (B[poz_7] * mask[6]) + (B[poz_8] * mask[7]) + (B[poz_9] * mask[8])) / s);
		G_new[index] = (int)(((G[poz_1] * mask[0]) + (G[poz_2] * mask[1]) + (G[poz_3] * mask[2]) + (G[poz_4] * mask[3]) + (G[poz_5] * mask[4]) + (G[poz_6] * mask[5]) + (G[poz_7] * mask[6]) + (G[poz_8] * mask[7]) + (G[poz_9] * mask[8])) / s);
		R_new[index] = (int)(((R[poz_1] * mask[0]) + (R[poz_2] * mask[1]) + (R[poz_3] * mask[2]) + (R[poz_4] * mask[3]) + (R[poz_5] * mask[4]) + (R[poz_6] * mask[5]) + (R[poz_7] * mask[6]) + (R[poz_8] * mask[7]) + (R[poz_9] * mask[8])) / s);
	

}

int main()
{

	header();
	

	char z;

	// deklaracja zmiennych
	int *B, *G, *R;
	long liczba_pikseli = Picture.biWidth*Picture.biHeight;
	cout << "Liczba pikseli: " << liczba_pikseli << endl;
	system("pause");

	B = new int[liczba_pikseli*sizeof(int)];
	G = new int[liczba_pikseli*sizeof(int)];
	R = new int[liczba_pikseli*sizeof(int)];


	cout << "--------------" << endl;
	fseek(forg, 0, SEEK_SET);
	for (int i = 0; i < File.bfOffBits; i++)
	{
		z = fgetc(forg);
		fprintf(fsz, "%c", z);                   //Utworzenie naglowka nowej Bitmapy
	}

	int licznik_znakow = 0;
	for (int i = File.bfOffBits; i < File.bfOffBits + liczba_pikseli; i++) //wczytanie pikseli
	{
		int index = i - File.bfOffBits;

		B[index] = (int)(fgetc(forg));
		G[index] = (int)(fgetc(forg));
		R[index] = (int)(fgetc(forg));
		licznik_znakow += 3;
		if (licznik_znakow == 3 * Picture.biWidth){
			int uzupelnienie_wiersza = (Picture.biSizeImage - 2 - 3 * liczba_pikseli) / Picture.biWidth;
			//cout << "Uzupelnienie wiersza: " << uzupelnienie_wiersza << endl;
			for (int i = 0; i < uzupelnienie_wiersza; i++){
				fgetc(forg);
			}
			licznik_znakow = 0;
		}
		//if (index > liczba_pikseli-475)
		//cout << index << ": " << "B: " << B[index] << " G: " << G[index] << " R: " << R[index] << endl;
	}

	system("pause");

	cudaError_t cudaStatus = GaussianBlurWithCuda(B, G, R, liczba_pikseli, Picture.biWidth);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "GaussianBlurWithCuda failed!\n");
		return 1;
	}

	cout << "--------------" << endl;



	/*for (int i = 0; i < File.bfOffBits; i++)
	{
		z = fgetc(forg);
		fprintf(fsz, "%c", z);                   //Utworzenie naglowka nowej Bitmapy
	}*/
	fseek(fsz, 54, SEEK_SET);
	licznik_znakow = 0;
	for (int i = 0; i < liczba_pikseli; i++)
	{
		fprintf(fsz, "%c", (int)(B[i]));
		fprintf(fsz, "%c", (int)(G[i]));
		fprintf(fsz, "%c", (int)(R[i]));
		licznik_znakow += 3;
		if (licznik_znakow == 3 * Picture.biWidth){
			int uzupelnienie_wiersza = (Picture.biSizeImage - 2 - 3 * liczba_pikseli) / Picture.biWidth;
			for (int i = 0; i < uzupelnienie_wiersza; i++){
				fprintf(fsz, "%c", (int)0);
			}
			licznik_znakow = 0;
		}
		//if (i > liczba_pikseli-475)
		//cout << i << ": " << "B: " << B[i] << " G: " << G[i] << " R: " << R[i] << endl;
	}
	fprintf(fsz, "%c", (int)0);
	fprintf(fsz, "%c", (int)0);

	delete[] B;
	delete[] G;
	delete[] R;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//system("PAUSE");
}

cudaError_t GaussianBlurWithCuda(int *b, int *g, int *r, long size, int width)
{
	int *d_B, *d_G, *d_R;
	int *d_B_new, *d_G_new, *d_R_new;
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

	cudaStatus = cudaMalloc((void**)&d_B_new, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_G, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_G_new, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_R_new, size * sizeof(int));
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
	GaussianBlur << < (size + 1023) / 1024, 1024 >> >(d_B, d_G, d_R, size, width, d_B_new, d_G_new, d_R_new);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "GaussianBlur launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching GaussianBlur!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(b, d_B_new, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(g, d_G_new, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(r, d_R_new, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_B);
	cudaFree(d_G);
	cudaFree(d_R);

	cudaFree(d_B_new);
	cudaFree(d_G_new);
	cudaFree(d_R_new);

	return cudaStatus;
}