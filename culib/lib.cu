#include "lib.cuh"
#include "cusolver_common.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include <vector>


namespace culib {
__host__ void make_kernel_param(dim3& grid_dim, dim3& block_dim, const dim3& num_tasks, int prefer_block_size) {
	block_dim.x = prefer_block_size;
	block_dim.y = prefer_block_size;
	block_dim.z = prefer_block_size;
	grid_dim.x = (num_tasks.x + prefer_block_size - 1) / prefer_block_size;
	grid_dim.y = (num_tasks.y + prefer_block_size - 1) / prefer_block_size;
	grid_dim.z = (num_tasks.z + prefer_block_size - 1) / prefer_block_size;
}

TempBuffer::TempBuffer(size_t size, bool unify /*= false*/) : unified(unify), siz(size) {
	if (!unify) {
		cudaMalloc(&pdata, size);
		cuda_error_check;
	}
	else {
		cudaMallocManaged(&pdata, size);
		cuda_error_check;
	}
}

TempBufferPlace::TempBufferPlace(TempBufferPool& pool, int bufid, std::unique_ptr<TempBuffer> tmp)
	: pool_(pool), bufferid(bufid), buffer(std::move(tmp)) { }
TempBufferPlace::TempBufferPlace(TempBufferPlace&& place)
	: pool_(place.pool_), bufferid(place.bufferid), buffer(std::move(place.buffer)) { }

TempBufferPlace::~TempBufferPlace(void) {
	if (buffer) pool_[bufferid] = std::move(buffer);
}

ManagedTempBlock::ManagedTempBlock(ManagedTempBlock&& other)
	: pool(other.pool), startBlock(other.startBlock), endBlock(other.endBlock) {
	endBlock = -1;
}

ManagedTempBlock::ManagedTempBlock(TempBufferPool& pool_, int startBlock_, int endBlock_)
	: pool(pool_), startBlock(startBlock_), endBlock(endBlock_)
{
	if (endBlock > pool.blockPlace32.size()) {
		printf("\033[31Preallocated unified buffer is not enough\033[0m\n");
		throw std::runtime_error("not enough unified memory");
	}
	for (int i = startBlock; i < endBlock; i++) {
		pool.blockPlace32[i] = true;
	}
}

ManagedTempBlock::~ManagedTempBlock(void) {
	for (int i = startBlock; i < endBlock; i++) {
		pool.blockPlace32[i] = false;
	}
}

TempBufferPool::TempBufferPool(void) {
	// preallocate enough temporary managed buffer, 
	int n32 = 1e4;
	unifiedBuffer.reset(new TempBuffer(32 * n32, true));
	blockPlace32.resize(n32, false);
}

TempBufferPlace TempBufferPool::getBuffer(size_t requireSize) {
	int matchid = -1;
	// find most matched buffer
	for (int i = 0; i < buffers.size(); i++) {
		if (buffers[i]) {
			if (buffers[i]->siz >= requireSize) {
				if (matchid<0 || buffers[matchid]->siz > buffers[i]->siz) {
					matchid = i;
				}
			}
		}
	}
	// if no match, allocate new
	if (matchid < 0) {
		buffers.emplace_back(std::make_unique<TempBuffer>(round(requireSize, 512)));
		matchid = buffers.size() - 1;
	}
	TempBufferPlace buf(*this, matchid, std::move(buffers[matchid]));
	return std::move(buf);
}

TempBufferPlace getTempBuffer(size_t siz) {
	return getTempPool().getBuffer(siz);
}

std::unique_ptr<TempBuffer>& TempBufferPool::operator[](size_t id) {
	if (id > buffers.size()) {
		printf("\033[31mBuffer id out of range\033[0m\n");
	}
	return buffers[id];
}

//TempBufferPool tempPool;

TempBufferPool& getTempPool(void) {
	static std::unique_ptr<TempBufferPool> Pool;
	if (!Pool) {
		Pool = std::make_unique<TempBufferPool>();
	}
	return *Pool; 
}

void show_cuSolver_version(void) {
#if 0
	int major = -1, minor = -1, patch = -1;
	cusolverGetProperty(MAJOR_VERSION, &major);
	cusolverGetProperty(MINOR_VERSION, &minor);
	cusolverGetProperty(PATCH_LEVEL, &patch);
	printf("[cuda version] : %d.%d.%d\n", major, minor, patch);
#else
#endif
}


void init_cuda(void) {
	get_device_info();
	//show_cuSolver_version();
	//if (std::is_same<Scaler, double>::value) {
	//	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	//	printf("[bank width] : set 8 bytes \n");
	//}
	//else if(std::is_same<Scaler, float>::value){
	//	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	//	printf("[bank width] : set 4 bytes \n");
	//}

	//printf("\n");

	//
	//printf("-- test library...\n");
	lib_test();
}

static int bankWidth = 0;

void use4Bytesbank(void) {
	if (bankWidth != 4) {
		bankWidth = 4;
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
		printf("\033[32m[CUDA] bank Width = 4\n\033[0m");
	}
}
void use8Bytesbank(void){
	if (bankWidth != 8) {
		bankWidth = 8;
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
		printf("\033[32m[CUDA] bank Width = 8\n\033[0m");
	}
}


void lib_test(void) {

}

int get_device_info()
{
	int device_count{ 0 };
	// get number of devices 
	cudaGetDeviceCount(&device_count);
	fprintf(stdout, "[GPU device Number]: %d\n", device_count);

	if (device_count == 0) {
		fprintf(stdout, "\033[31mNo CUDA supported device Found!\033[0m\n");
		exit(-1);
	}

	int usedevice = 0;

	cudaDeviceProp use_device_prop;
	use_device_prop.major = 0;
	use_device_prop.minor = 0;

	fprintf(stdout, "- Enumerating Device...\n");
	for (int dev = 0; dev < device_count; ++dev) {
		fprintf(stdout, "---------------------------------------------------------------\n");
		int driver_version{ 0 }, runtime_version{ 0 };
		// set cuda execuation GPU 
		cudaSetDevice(dev);
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, dev);

		fprintf(stdout, "\n[Device %d]: %s\n", dev, device_prop.name);

		cudaDriverGetVersion(&driver_version);
		fprintf(stdout, "[CUDA driver]:- - - - - - - - - - - - - - - - - %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
		cudaRuntimeGetVersion(&runtime_version);
		fprintf(stdout, "[CUDA runtime]: - - - - - - - - - - - - - - - - %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
		fprintf(stdout, "[Device Capicity]: - -  - - - - - - - - - - - - %d.%d\n", device_prop.major, device_prop.minor);

		if (device_prop.major >= use_device_prop.major && device_prop.minor >= use_device_prop.minor) {
			usedevice = dev;
			use_device_prop = device_prop;
		}
	}
	fprintf(stdout, " \n");
	fprintf(stdout, "- set device %d [%s]\n", usedevice, use_device_prop.name);
	cudaSetDevice(usedevice);

	return 0;
}

}

