#include "DeviceBuffer.h"
#include "regex"

namespace homo {

	//BufferManager mem;

	std::string getMemType(MemType type_)
	{
		switch (type_) {
		case Device:
			return "Device";
		case Pinned:
			return "Pinned";
		case Managed:
			return "Managed";
		case Hostheap:
			return "Hostheap";
		}
		return "";
	}

	static std::unique_ptr<BufferManager> pmem;

	BufferManager& getMem() {
		if (!pmem) {
			pmem = std::make_unique<BufferManager>();
		}
		return *pmem;
	}

	void freeMem(void) { pmem.reset(); }

	DeviceBuffer::~DeviceBuffer()
	{
		if (data_ != nullptr) {
			cudaError_t err = cudaSuccess;
			switch (mType) {
			case Device:
				err = cudaFree(data_);
				break;
			case Pinned:
				err = cudaFreeHost(data_);
				break;
			case Managed:
				err = cudaFree(data_);
				break;
			case Hostheap:
				try {
					free(data_);
					err = cudaSuccess;
				}
				catch (...) {
					err = cudaErrorInvalidDevicePointer;
				}
				break;
			}
			if (err) {
				printf("\033[31mFree buffer failed with err %s, memType %s\033[0m\n", cudaGetErrorName(err), getMemType(mType).c_str());
				throw std::runtime_error("failed to Free memory");
			}
		}
	}

	DeviceBuffer::DeviceBuffer(size_t size, MemType memType /*= Device*/) : mType(memType), size_(size)
	{
		cudaError_t err = cudaErrorUnknown;
		// DEBUG
		{
			//size_t freeMem, totalMem;
			//cudaMemGetInfo(&freeMem, &totalMem);
			//printf("Free mem %dMB, totalMem %dMB\n", (int)(freeMem / 1024 / 1024), (int)(totalMem / 1024 / 1024));
			//cudaDeviceSynchronize();
			//err = cudaPeekAtLastError();
			//if (err) {
			//	printf("\033[31mCuda error detected %s before allocating buffer\033[0m\n", cudaGetErrorName(err));
			//	throw std::runtime_error("failed to allocate GPU memory");
			//}
		}
		switch (memType)
		{
		case Device:
			err = cudaMalloc(&data_, size);
			break;
		case Pinned:
			err = cudaMallocHost(&data_, size);
			break;
		case Managed:
			err = cudaMallocManaged(&data_, size);
			break;
		case Hostheap:
			try {
				data_ = malloc(size);
				err = cudaSuccess;
			}
			catch (...) {
				err = cudaErrorMemoryAllocation;
			}
			break;
		default:
			break;
		}
		if (err) {
			printf("\033[31mcudaMalloc Failed with err %s, required memory %dMB, memType %s\033[0m\n",
				cudaGetErrorName(err), (int)(size / 1024 / 1024), getMemType(mType).c_str());
			throw std::runtime_error("failed to allocate GPU memory");
		}
	}

	BufferManager::BufferManager(void)
	{
		cudaGetDevice(&devId);
	}

	size_t BufferManager::size(std::string namepattern) const
	{
		size_t memsize = 0;
		std::regex reg(namepattern);
		for (auto iter = buffers.begin(); iter != buffers.end(); iter++) {
			std::string bufname = iter->first;
			bool matched = std::regex_match(bufname, reg);
			if (matched) memsize += iter->second->size();
		}
		return memsize;
	}

}
