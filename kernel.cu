/*
andrew's cuda code 10.09.2020
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <chrono>
using namespace std;

#define ll  long long int


const ll m = 0x5DEECE66Dll;
const ll mask = (1ll << 48) - 1;

#define advance1(s) s = (s * m + 11ll) & mask
#define advance3759(s) s = (s*0x6fe85c031f25ll + 0x8f50ecff899ll)&mask
#define advance16(s) s = (s*0x6dc260740241ll + 0xd0352014d90ll)&mask
#define advance387(s) s = (s*0x5fe2bcef32b5ll + 0xb072b3bf0cbdll)&mask
#define advance774(s) s = (s*0xf8d900133f9ll + 0x5738cac2f85ell)&mask
#define advance11(s) s = (s*0x53bce7b8c655ll + 0x3bb194f24a25ll)&mask
#define advance3(s) s = (s*0xd498bd0ac4b5ll + 0xaa8544e593dll)&mask
#define advance17(s) s = (s*0xee96bd575badll + 0xc45d76fd665bll)&mask

#define regress1(s) s = (s*0xdfe05bcb1365ll + 0x615c0e462aa9ll)&mask
#define regress3(s) s = (s*0x13a1f16f099dll + 0x95756c5d2097ll)&mask
#define regress3759(s) s = (s*0x63a9985be4adll + 0xa9aa8da9bc9bll)&mask
#define advance830(s) s = (s*0x859d39e832d9ll + 0xe3e2df5e9196ll)&mask
#define advance3760(s) s = (s*0x8c35c76b80c1ll + 0xd7f102f24f30ll)&mask
#define regress387(s) s = (s*0x6f629b13ab9dll + 0x5458c54bf117ll)&mask


#define getNextInt(x, s) advance1(s); x = (int)(s>>16)

// need spare longs temp1 and temp2
#define getNextLong(x, s) getNextInt(temp1, s); getNextInt(temp2, s); x = (temp1 << 32) + temp2

#//define getIntBounded(x, s, n) if ((n&(-n))==n) {advance1(s); x = (int)((n*(s>>17)) >> 31);} else {do{advance1(s); bits = s>>17; val = bits%n;}while(bits-val+(n-1)<0); x=val;}
#define getIntBounded(x, s, n) advance1(s); x = (int)((n*(s>>17)) >> 31);

#define setSeed(s, x) s = (x^m)&mask

const int STATUS_MASK = ((1 << 18) - 1) << 8;

__device__ __managed__ unsigned long long int num_found = 0;
#define memsz 10000
__device__ __managed__ ll ret[memsz];
__device__ __managed__ ll retx[memsz];
__device__ __managed__ ll rety[memsz];
__device__ __managed__ ll retz[memsz];
__device__ __managed__ ll retn[memsz];
__device__ __managed__ int table[17][18];


#define seg(sx, ex, z, y) for (int i = sx; i < ex; i++) {table[i][z] = y;q++;}
#define flower(x, z) table[x][z] |= (1<<(num_flowers+++8))

void init_table() {
	for (int x = 0; x < 17; x++) {
		for (int z = 0; z < 18; ++z)
		{
			table[x][z] = 3;
		}
	}
	int q = 0;
	seg(0, 16, 0, 70); seg(16, 17, 0, 69);
	seg(0, 14, 1, 70); seg(14, 17, 1, 69);
	seg(0, 12, 2, 70); seg(12, 17, 2, 69);
	seg(0, 8, 3, 70); seg(8, 17, 3, 69);
	seg(0, 17, 4, 69);
	seg(0, 17, 5, 69);
	seg(0, 1, 6, 69); seg(1, 2, 6, 68); seg(2, 17, 6, 69);
	seg(0, 5, 7, 68); seg(5, 14, 7, 69); seg(14, 17, 7, 68);
	seg(0, 9, 8, 68); seg(9, 10, 8, 69); seg(10, 17, 8, 68);
	seg(0, 17, 9, 68);
	seg(0, 3, 10, 67); seg(3, 15, 10, 68); seg(15, 17, 10, 67);
	seg(0, 5, 11, 67); seg(5, 12, 11, 68); seg(12, 17, 11, 67);
	seg(0, 6, 12, 67); seg(6, 11, 12, 68); seg(11, 14, 12, 67); seg(14, 17, 12, 66);
	seg(0, 7, 13, 67); seg(7, 10, 13, 68); seg(10, 13, 13, 67); seg(13, 16, 13, 66); seg(16, 17, 13, 65);
	seg(0, 2, 14, 66); seg(2, 9, 14, 67); seg(9, 10, 14, 68); seg(10, 12, 14, 67); seg(12, 13, 14, 66); seg(13, 16, 14, 65); seg(16, 17, 14, 64);
	seg(0, 3, 15, 66); seg(3, 11, 15, 67); seg(11, 12, 15, 66); seg(12, 14, 15, 65); seg(14, 17, 15, 64);
	seg(0, 4, 16, 66); seg(4, 10, 16, 67); seg(10, 12, 16, 66); seg(12, 14, 16, 65); seg(14, 17, 16, 64);
	seg(0, 9, 17, 66); seg(9, 10, 17, 67); seg(10, 11, 17, 66); seg(11, 17, 17, 33);
	seg(12, 15, 16, 11); // more suspicious 
	seg(13, 15, 15, 11); // ^
	for (int z = 0; z < 18; z++) {
		for (int x = 0; x < 17; ++x)
		{
			printf("%d ", table[x][z]);
		}
		printf("\n");
	}
	printf("%d\n", q);

	int num_flowers = 0;
	flower(2, 11);
	flower(3, 10);
	flower(4, 3);
	flower(4, 8);
	flower(5, 12);
	flower(6, 6);
	flower(6, 10);
	flower(7, 12);
	flower(8, 8);
	flower(8, 12);
	flower(8, 13);
	flower(9, 7);
	flower(9, 10);
	flower(9, 11);
	flower(9, 14);
	flower(11, 8);
	flower(11, 14);
	flower(14, 9);
	int check = 0;
	for (int z = 0; z < 18; z++) {
		for (int x = 0; x < 17; ++x)
		{
			printf("%d ", table[x][z] & STATUS_MASK);
			check |= table[x][z] & STATUS_MASK;
		}
		printf("\n");
	}
	printf("check_ok %d\n", check == STATUS_MASK);

}


__device__ void output_seed(int n, int x, int y, int z, ll s) {
	//printf("writing: %llu \n", s);
	ll id = atomicAdd(&num_found, 1ull); // dw about red underline
	ret[id] = s;
	retx[id] = x;
	rety[id] = y;
	retz[id] = z;
	retn[id] = n;
}


__device__ __managed__ int max_found = 0;
__device__ void checkParall(ll s, int y, int z, int s_table[17][18]) {
	int a, b, fx, fy, fz;
	int status[3][4] = { 0 }; // parity: continue or not ;
	char found[3][4] = {0};
	
	int eliminated = 0;
	for (int i = 0; i < 64; ++i)
	{
		getIntBounded(a, s, 8); getIntBounded(b, s, 8);
		fx = a - b;
		getIntBounded(a, s, 4); getIntBounded(b, s, 4);
		fy = y + a - b;
		getIntBounded(a, s, 8); getIntBounded(b, s, 8);
		fz = a - b;


		for (int x = 0; x < 3; x++) {
			//for (int z = 0; z < 4; z++) {
				if (status[x][z] & 1)
				{
					continue;
				}

				int look_up = s_table[fx + x + 7][fz + z + 7];
				int look_up_y = look_up & 127;
				if (look_up_y == fy)
				{
					int flower_status = STATUS_MASK & look_up;
					if (flower_status)
					{
						if ((status[x][z] | flower_status) != status[x][z]) {
							status[x][z] |= flower_status;
							found[x][z]++;
						}
					}
					else {
						status[x][z] |= 1;
						eliminated++;
					
						if (eliminated == 3)
						{
							return;
						}
					}
				}
			//}
		}
	}
	for (int x = 0; x < 3; x++) {
		//for (int z = 0; z < 4; z++) {
			if (found[x][z] >= 9 && (!(status[x][z]&1))) { // note : last bit of status is not allowed to a 1
				//max_found = found[x][z];
				regress387(s);
				//printf("found = %d\n", found[x][z]);
				output_seed(found[x][z], x, y, z, s);
			}
		//}
	}
}

__device__ int foo() {
	return 70;
}



__global__ void searchKernel2(ll o, ll y) {



	ll input_seed = blockDim.x * blockIdx.x + threadIdx.x + o;
	__shared__ int s_table[17][18];
	if (threadIdx.x < 306) {
		int i = threadIdx.x % 17;
		int j = threadIdx.x / 17;
		s_table[threadIdx.x % 17][threadIdx.x / 17] = table[threadIdx.x % 17][threadIdx.x / 17];
	}
	
	/*
	if (threadIdx.x == 0) {
			for (int i = 0; i < 17; i++) {
				for (int j = 0; j < 18; j++) {
					s_table[i][j] = table[i][j];
				}
			}
		
	}*/

	__syncthreads();
	/*
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		for (int i = 0; i < 17; i++) {
			for (int j = 0; j < 18; j++) {
				if (table[i][j] != s_table[i][j]) {
					printf("oh dear %d %d %d %d\n", i, j, s_table[i][j], table[i][j]);
				}
			}
		}
	}*/
	
	int z;
	ll seed;
	for (ll y = 67; y <= 69; y++) {
		if (y!=68) {
			continue; // most likely y=68
		}
		seed = (y << 41) + input_seed;
		getIntBounded(z, seed, 16);
		if (z >= 1 && z <= 4) {
			checkParall(seed, y, z - 1, s_table);
		}
	}
	
}








cudaError_t search_coords() {

	ofstream fout("flowers5.3.txt");
	ofstream log("log5.3.txt");
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}




	int threads_per_block = 512;
	int num_blocks = 32768;

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	//ll num_found = 0;
	printf("begin xyz\n");
	auto start = chrono::steady_clock::now();
	int num_written = 0;
	for (ll o = 0; o < (1ll << 41); o += threads_per_block * num_blocks) {
		searchKernel2 << <num_blocks, threads_per_block >> > (o, 68ll);
		if (o % (1ll << 30) == 0) {
			cudaDeviceSynchronize();
			for (int i = 0; i < num_found; i++) {
				fout << ret[i] << " " << retx[i] << " " << rety[i] << " " << retz[i] << " " << retn[i] << endl;
				num_written++;
			}
			num_found = 0;
			//printf("%lld\n", o);
			auto end = chrono::steady_clock::now();
			ll time = (chrono::duration_cast<chrono::microseconds>(end - start).count());
			float eta = (((1ll << 41) - o) / ((float)o)) * ((float)time) / 3600.0 / 1000000.0; 
			log << "doing " << o << " time taken us =" << time << " eta (hrs) = " << eta << endl;
			log.flush();
			fout.flush();
		}

	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cuda not sync: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	auto end = chrono::steady_clock::now();
	cout << "time taken us =" << chrono::duration_cast<chrono::microseconds>(end - start).count() << endl;


	printf(" num_found = %llu \n", num_found);
	//printf("(z=%d) num_found = %llu \n tc = %llu", z, num_found, tasks_completed);


	for (int i = 0; i < num_found; i++) {
		fout << ret[i] << " " << retx[i] << " " << rety[i] << " " << retz[i] << endl;
		num_written++;
	}
	cout << "total seeds written=" << num_written << endl;

	//}

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

Error:
	return cudaStatus;
}

int main()
{
	init_table();
	// Add vectors in parallel.
	cudaError_t cudaStatus = search_coords();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
