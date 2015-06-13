#include <windows.h>
#include <time.h>
#include <cstdlib>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

cudaError_t addWithCuda(int *b, int *g, int *r, long size, int width);

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
	cout << "Liczba bitow na piksel:  (1, 4, 8, or 24)" << Picture.biBitCount << endl;

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

__global__ void ReadImage(int *B, int *G, int *R, int numberOfPixels, int width, int *B_new, int *G_new, int *R_new)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index < width){ // dolny rzad pikseli
		B_new[index] = B[index];
		G_new[index] = G[index];
		R_new[index] = R[index];
		return;
	}
	if (index > numberOfPixels - width){ //gorny rzad pikseli
		B_new[index] = B[index];
		G_new[index] = G[index];
		R_new[index] = R[index];
		return;
	}
	if (index % width == 0){ //lewa sciana
		B_new[index] = B[index];
		G_new[index] = G[index];
		R_new[index] = R[index];
		return;
	}
	if (index % width == width - 1){ //prawa sciana
		B_new[index] = B[index];
		G_new[index] = G[index];
		R_new[index] = R[index];
		return;
	}

	int mask[] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	int s = 16;
	
	if (s != 0 && index < numberOfPixels)
	{
		

		int poz_1 = index - width - 1;
		int poz_2 = index - width;
		int poz_3 = index - width + 1;
		int poz_4 = index - 1;
		int poz_5 = index;
		int poz_6 = index + 1;
		int poz_7 = index + width - 1;
		int poz_8 = index + width;
		int poz_9 = index + width + 1;

		B_new[index] =  (int)( ((B[poz_1] * mask[0]) + (B[poz_2] * mask[1]) + (B[poz_3] * mask[2]) + (B[poz_4] * mask[3]) + (B[poz_5] * mask[4]) + (B[poz_6] * mask[5]) + (B[poz_7] * mask[6]) + (B[poz_8] * mask[7]) + (B[poz_9] * mask[8])) /s);
		G_new[index] = (int)( ((G[poz_1] * mask[0]) + (G[poz_2] * mask[1]) + (G[poz_3] * mask[2]) + (G[poz_4] * mask[3]) + (G[poz_5] * mask[4]) + (G[poz_6] * mask[5]) + (G[poz_7] * mask[6]) + (G[poz_8] * mask[7]) + (G[poz_9] * mask[8])) /s );
		R_new[index] = (int)( ((R[poz_1] * mask[0]) + (R[poz_2] * mask[1]) + (R[poz_3] * mask[2]) + (R[poz_4] * mask[3]) + (R[poz_5] * mask[4]) + (R[poz_6] * mask[5]) + (R[poz_7] * mask[6]) + (R[poz_8] * mask[7]) + (R[poz_9] * mask[8])) /s);
	}

}

int main()
{

	header();
	char* filename = "picasso_123.bmp";
	char z;

	int i;
	FILE* f = fopen(filename, "rb");

	if (f == NULL)
		throw "Argument Exception";

	unsigned char info[54];
	fread(info, sizeof(unsigned char), 54, f); // read the 54-byte header

	// extract image height and width from header
	int width = Picture.biWidth;
	int height = Picture.biHeight;

	//cout << endl;
	//cout << "  Name: " << filename << endl;
	//cout << " Width: " << width << endl;
	//cout << "Height: " << height << endl;

	int row_padded = (width * 3 + 3) & (~3);
	unsigned char* data = new unsigned char[row_padded];
	unsigned char tmp;

	// deklaracja zmiennych
	int *B, *G, *R;
	long liczba_pikseli = width*height;

	B = new int[liczba_pikseli*sizeof(int)];
	G = new int[liczba_pikseli*sizeof(int)];
	R = new int[liczba_pikseli*sizeof(int)];

	/*long licznik_pikseli = 0;
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

			//cout << licznik_pikseli << ": " << "R: " << R[licznik_pikseli] << " G: " << G[licznik_pikseli] << " B: " << B[licznik_pikseli] << endl;
			licznik_pikseli++;
		}
	}

	fclose(f);*/

	cout << "--------------" << endl;
	fseek(forg, 0, SEEK_SET);
	for (int i = 0; i<File.bfOffBits; i++)
	{
		z = fgetc(forg);
		fprintf(fsz, "%c", z);                   //Utworzenie naglowka nowej Bitmapy
	}
	
	int licznik_znakow = 0;
	for (int i = File.bfOffBits; i < File.bfOffBits+liczba_pikseli; i++) //wczytanie pikseli
	{
		B[i] = (int)(fgetc(forg));
		licznik_znakow++;
		if (licznik_znakow == 3*width){
			cout << "znak nadmiarowy: " << (int)fgetc(forg) << endl;
			cout << "znak nadmiarowy: " << (int)fgetc(forg) << endl;
			licznik_znakow = 0;
		}
		G[i] = (int)(fgetc(forg));
		licznik_znakow++;
		if (licznik_znakow == 3*width){
			cout << "znak nadmiarowy: " << (int)fgetc(forg) << endl;
			cout << "znak nadmiarowy: " << (int)fgetc(forg) << endl;
			licznik_znakow = 0;
		}
		R[i] = (int)(fgetc(forg));
		licznik_znakow++;
		if (licznik_znakow == 3*width){
			cout << "znak nadmiarowy: " << (int)fgetc(forg) << endl;
			cout << "znak nadmiarowy: " << (int)fgetc(forg) << endl;
			licznik_znakow = 0;
		}
		cout << i - File.bfOffBits << ": " << "B: " << B[i] << " G: " << G[i] << " R: " << R[i] << endl;
	}

	cudaError_t cudaStatus = addWithCuda(B, G, R, liczba_pikseli, width);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!\n");
		return 1;
	}

	cout << "--------------" << endl;

	for (int i = 0; i < liczba_pikseli; i++){
		//cout << i << ": " << "R: " << R[i] << " G: " << G[i] << " B: " << B[i] << endl;
	}


	for (int i = 0; i < File.bfOffBits; i++)
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

	for (int i = File.bfOffBits + liczba_pikseli; i < File.bfSize; i++){
		fprintf(fsz, "%c", (int)fgetc(forg));
	}

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

cudaError_t addWithCuda(int *b, int *g, int *r, long size, int width)
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
	ReadImage << <1, size >> >(d_B, d_G, d_R, size, width, d_B_new, d_G_new, d_R_new);

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

	return cudaStatus;
}