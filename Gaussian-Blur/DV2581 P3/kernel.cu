#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CImg.h"
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include <vector>

using namespace cimg_library;

// Declaring the relevant data globally.
#define WIDTH 4000
#define HEIGHT 3000
#define SPECTRUM 3
#define MASK_SIZE 5
#define PIXELS (WIDTH*HEIGHT)
#define VALUES (PIXELS*SPECTRUM)

double *mask = new double[MASK_SIZE*MASK_SIZE];

unsigned char *pic;

struct kernelInfo
{
	int n_threads; // Threads per block.
	int n_blocks; // Number of blocks.
	int per_thread; // Number of actions per thread.
};
kernelInfo info;

void getKernelInfo();

void blur_with_cuda(int block_size, int grid_size);

void restructureCImg(bool toPresent);

double checkBlur(CImg<unsigned char> i1, CImg<unsigned char> i2);

__device__
int index(int row, int col);

__global__
void blur(unsigned char *input, unsigned char *output, double *mask, int per_thread);

__global__
void structureByColor(unsigned char *input, unsigned char *output, int per_thread);

__global__
void structureByPixel(unsigned char *input, unsigned char *output, int per_thread);


int main()
{
	std::ofstream out;
	out.open("results.txt");

	CImg<unsigned char> image("cake.ppm"), blurImage("cake.ppm");
	pic = image.data();

	CImgDisplay disp(image, "Original image");
	disp.resize(640, 480);
	disp.move(0, disp.window_y());
	// Performing and timing the sequential blur.
	CImg<double> mask5(5, 5);
	mask5(0, 0) = mask5(0, 4) = mask5(4, 0) = mask5(4, 4) = 1.0 / 256.0;
	mask5(0, 1) = mask5(0, 3) = mask5(1, 0) = mask5(1, 4) = mask5(3, 0) = mask5(3, 4) = mask5(4, 1) = mask5(4, 3) = 4.0 / 256.0;
	mask5(0, 2) = mask5(2, 0) = mask5(2, 4) = mask5(4, 2) = 6.0 / 256.0;
	mask5(1, 1) = mask5(1, 3) = mask5(3, 1) = mask5(3, 3) = 16.0 / 256.0;
	mask5(1, 2) = mask5(2, 1) = mask5(2, 3) = mask5(3, 2) = 24.0 / 256.0;
	mask5(2, 2) = 36.0 / 256.0;
	
	// Convolve and record the time taken to do the operation
	auto begin_CPU = std::chrono::high_resolution_clock::now();
	// Blur the image!
	blurImage.convolve(mask5);
	auto end_CPU = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_CPU = end_CPU - begin_CPU;
	std::cout << "Time taken to convolve on CPU = " << elapsed_CPU.count() << " seconds.\n";

	CImgDisplay blur_disp_CPU(blurImage, "Blurred image on CPU");
	blur_disp_CPU.resize(640, 480);
	blur_disp_CPU.move(640, blur_disp_CPU.window_y());

	// Filling the info struct with kernel launch parameters
	// based on the device used.
	getKernelInfo();

	// Restructuring the pixel buffer to RGBRGBRGB... 
	// to fit the instructions for the project.
	restructureCImg(false);

	// Filling the mask as requested.
	mask[0*MASK_SIZE + 0] = mask[0*MASK_SIZE + 4] = mask[4*MASK_SIZE + 0] = mask[4*MASK_SIZE + 4] = 1.0 / 256.0;
	mask[0*MASK_SIZE + 1] = mask[0*MASK_SIZE + 3] = mask[1*MASK_SIZE + 0] = mask[1*MASK_SIZE + 4] = mask[3*MASK_SIZE + 0] = mask[3*MASK_SIZE + 4] = mask[4*MASK_SIZE + 1] = mask[4*MASK_SIZE + 3] = 4.0 / 256.0;
	mask[0*MASK_SIZE + 2] = mask[2*MASK_SIZE + 0] = mask[2*MASK_SIZE + 4] = mask[4*MASK_SIZE + 2] = 6.0 / 256.0;
	mask[1*MASK_SIZE + 1] = mask[1*MASK_SIZE + 3] = mask[3*MASK_SIZE + 1] = mask[3*MASK_SIZE + 3] = 16.0 / 256.0;
	mask[1*MASK_SIZE + 2] = mask[2*MASK_SIZE + 1] = mask[2*MASK_SIZE + 3] = mask[3*MASK_SIZE + 2] = 24.0 / 256.0;
	mask[2*MASK_SIZE + 2] = 36.0 / 256.0;
	
	struct resultInfo
	{
		int grid_size;
		int block_size;
		double time;
	};
	std::vector<resultInfo> result;
	resultInfo fastest;
	fastest.time = 1000.0;
	for (int grid_size = 30; grid_size <= 3000; grid_size += 30)
	{
		for (int block_size = 256; block_size <= 1024; block_size *= 2)
		{
			auto begin_GPU = std::chrono::high_resolution_clock::now();	
			blur_with_cuda(block_size, grid_size);	
			auto end_GPU = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_GPU = end_GPU - begin_GPU;
			printf("%i blocks with %i threads each completed in %5.7f seconds.\n", grid_size, block_size, elapsed_GPU.count());

			resultInfo r;
			r.block_size = block_size;
			r.grid_size = grid_size;
			r.time = elapsed_GPU.count();
			result.push_back(r);

			if (r.time < fastest.time)
			{
				fastest = r;
			}
		}
	}

	for (int i = 256; i <= 1024; i *= 2)
	{
		out << i << std::endl;
		for (int j = 0; j < result.size(); j++)
		{
			if (result[j].block_size == i)
			{
				out << result[j].grid_size << "\t" << result[j].time << std::endl;
			}
		}
	}

	out << "The fastest recorded time was " << fastest.time << " with " << fastest.grid_size << " blocks of " << fastest.block_size << " threads each.\n";

	// Restructuring the pixel buffer back to RRR...GGG...BBB...
	// in order to present the picture easily.
	restructureCImg(true);

	CImg<unsigned char> blurred_image_GPU(pic, WIDTH, HEIGHT, 1, SPECTRUM);
	CImgDisplay blur_disp_GPU(blurred_image_GPU, "Blurred image on GPU");
	blur_disp_GPU.resize(640, 480);
	blur_disp_GPU.move(1280, blur_disp_GPU.window_y());

	// Checking how close the results are. Somewhat large difference likely comes from the handling of edge values.
	std::cout << "CPU and GPU color values have a difference of " << checkBlur(blurImage, blurred_image_GPU) << "on average." << std::endl;


	delete[] mask;
	out.close();
	getchar();
	return 0;
}

void getKernelInfo()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	unsigned int max_total_threads, n_blocks, n_threads, per_thread;	

	max_total_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
	per_thread = (int)std::ceil((double)VALUES / (double)max_total_threads);

	// Want to fit a number of blocks perfectly.
	n_threads = prop.maxThreadsPerBlock;
	while (prop.maxThreadsPerMultiProcessor % n_threads != 0)
	{
		n_threads /= 2;
	}

	// Calculating the number of blocks needed.
	n_blocks = (int)std::ceil((double)VALUES / (double)(n_threads * per_thread));

	// Padding the block count to make it divisible by the number of colors.
	// This is to have the same number of blocks working on each color.
	while (n_blocks % SPECTRUM != 0)
	{
		n_blocks++;
	}

	// Storing these values into the global info struct.
	info.n_blocks = n_blocks;
	info.n_threads = n_threads;
	info.per_thread = per_thread;
}

void blur_with_cuda(int block_size, int grid_size)
{
	// Checking so that the parameters are valid to
	// launch a kernel with.
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	if (block_size > prop.maxThreadsPerBlock)
	{
		std::cout << "Error: Block size cannot be greater than " << prop.maxThreadsPerBlock << std::endl;
		return;
	}
	if (grid_size % SPECTRUM != 0)
	{
		std::cout << "Error: Grid size must be divisible by " << SPECTRUM << std::endl;
		return;
	}

	// Calculates how much needs to be done by each thread based on the provided
	// grid size and block size.
	int per_thread = (int)std::ceil((double)VALUES / (double)(grid_size*block_size));

	unsigned char *d_input, *d_output;
	double *d_mask;
	cudaMalloc(&d_input, VALUES * sizeof(unsigned char));
	cudaMalloc(&d_output, VALUES * sizeof(unsigned char));
	cudaMalloc(&d_mask, MASK_SIZE*MASK_SIZE * sizeof(double));

	// Copying the mask and picture over to the device.
	cudaMemcpy(d_input, pic, VALUES * sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, MASK_SIZE*MASK_SIZE * sizeof(double), cudaMemcpyHostToDevice);

	// Structuring the pixel buffer from RGBRGBRGB... to RRR...GGG...BBB...
	structureByColor<<< grid_size, block_size >>>(d_input, d_output, per_thread);

	// Blurring the image. Writing to the input from the output as the output holds the restructured image.
	blur<<< grid_size, block_size >>>(d_output, d_input, d_mask, per_thread);

	// Restructuring d_input into d_output as d_input will now hold the blurred image.
	structureByPixel<<< grid_size, block_size >>>(d_input, d_output, per_thread);
	
	// Copying the data from d_output back to the device.
	cudaMemcpy(pic, d_output, VALUES * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_mask);
}

void restructureCImg(bool toPresent)
{
	// Only used prior to and after timing of the actual
	// task, due to CImg already storing the pixel buffer 
	// in the desired format.

	// toPresent is used to tell the function whether it
	// should restructure the image to present (RRGGBB)
	// or not (RGBRGB).
	unsigned char *d_input, *d_output;
	
	cudaMalloc(&d_input, VALUES * sizeof(unsigned char));
	cudaMalloc(&d_output, VALUES * sizeof(unsigned char));
	cudaMemcpy(d_input, pic, VALUES * sizeof(unsigned char), cudaMemcpyHostToDevice);

	if (toPresent)
	{
		structureByColor << < info.n_blocks, info.n_threads >> > (d_input, d_output, info.per_thread);
	}
	else
	{
		structureByPixel << < info.n_blocks, info.n_threads >> > (d_input, d_output, info.per_thread);
	}

	cudaMemcpy(pic, d_output, VALUES * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}

double checkBlur(CImg<unsigned char> i1, CImg<unsigned char> i2)
{
	unsigned char* data1 = i1.data();
	unsigned char* data2 = i2.data();

	double avg_diff = 0.0;

	for (int i = 0; i < WIDTH*HEIGHT*SPECTRUM; i++)
	{
		avg_diff += std::abs(data1[i] - data2[i]);
	}
	avg_diff /= WIDTH * HEIGHT*SPECTRUM;

	return avg_diff;
}

__global__
void blur(unsigned char *input, unsigned char *output, double *mask, int per_thread)
{
	// Assumes pixel buffer structured by color rather than pixel.
	
	// Threads read values in sequence.
	int ti = threadIdx.x + per_thread * blockDim.x * blockIdx.x;
	for (int i = 0; i < per_thread; i++)
	{
		int out_index = ti + i * blockDim.x;

		// If this index is out of bounds, every following
		// index will be as well, so break.
		if (out_index >= VALUES) break;

		// Transforming the index into row and column
		// indices to be able to get surrounding 
		// elements more smoothly.
		int col = out_index % WIDTH;
		int row = (out_index - col) / WIDTH;

		// Taking every color value in a MASK_SIZE by MASK_SIZE area with
		// element (col,row) as the center, multiplying the color value
		// by the corresponding value in the mask and adding the result 
		// to result.
		double result = 0.0;		
		int curr_row, curr_col;	
		for (int j = -(MASK_SIZE - 1) / 2; j <= (MASK_SIZE - 1) / 2; j++)
		{
			curr_row = row + j;
			if (curr_row >= 0 && curr_row < HEIGHT*SPECTRUM)
			{
				for (int k = -(MASK_SIZE - 1) / 2; k <= (MASK_SIZE - 1) / 2; k++)
				{
					curr_col = col + k;
					if (curr_col >= 0 && curr_col < WIDTH)
					{
						result += mask[(j + (MASK_SIZE - 1) / 2)*MASK_SIZE + k + (MASK_SIZE - 1) / 2] * (double)input[index(curr_row, curr_col)];
					}
				}
			}
		}		
		// Replacing the color value in output with the resulting color value.
		// No need to reformat to unsigned char here as the input values used in
		// calculating the result are unsigned chars.
		output[out_index] = result;
	}
}

__global__
void structureByColor(unsigned char *input, unsigned char *output, int per_thread)
{
	// Takes RGBRGBRGB... and turns into RRR...GGG...BBB...

	// For every three blocks there should be one for each color.
	int colorIndex = blockIdx.x % SPECTRUM;
	
	// A third of the blocks work on one color, this determines
	// which index the block has for that specific color.
	int blockIndexForColor = (blockIdx.x - colorIndex) / SPECTRUM;

	int inputStart, outputStart;

	// Cleverly indexing so that elements are read optimally for memory coalescence.
	inputStart = SPECTRUM * per_thread *(blockIndexForColor*blockDim.x + threadIdx.x) + colorIndex;
	outputStart = per_thread * (blockIndexForColor*blockDim.x + threadIdx.x) + colorIndex * PIXELS;

	int inputIndex, outputIndex;
	for (int i = 0; i < per_thread; i++)
	{
		inputIndex = inputStart + i * SPECTRUM;
		outputIndex = outputStart + i;
		if (inputIndex < VALUES && outputIndex < VALUES)
		{
			output[outputIndex] = input[inputIndex];
		}
	}
}

__global__
void structureByPixel(unsigned char *input, unsigned char *output, int per_thread)
{
	// Takes RRR...GGG...BBB... and turns into RGBRGBRGB...
	int colorIndex = blockIdx.x % SPECTRUM;

	// A third of the blocks work on one color, this determines
	// which index the block has for that specific color.
	int blockIndexForColor = (blockIdx.x - colorIndex) / SPECTRUM;

	// Cleverly indexing so that elements are read optimally for memory coalescence.
	int inputStart = per_thread * (blockIndexForColor*blockDim.x + threadIdx.x) + colorIndex * PIXELS;
	int outputStart = SPECTRUM * per_thread *(blockIndexForColor*blockDim.x + threadIdx.x) + colorIndex;

	int inputIndex, outputIndex;
	for (int i = 0; i < per_thread; i++)
	{
		inputIndex = inputStart + i;
		outputIndex = outputStart + i * SPECTRUM;
		if (inputIndex < VALUES && outputIndex < VALUES)
		{
			output[outputIndex] = input[inputIndex];
		}
	}
}

__device__
int index(int row, int col)
{
	return row * WIDTH + col;
}