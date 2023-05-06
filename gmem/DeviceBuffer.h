#pragma once

#include "platform_spec.h"
#include <memory>
#include <stdexcept>
#include "cuda_runtime.h"
#include <string>
#include <map>

namespace homo {
	enum MemType {
		Device,
		Pinned,
		Managed,
		Hostheap
	};

	std::string getMemType(MemType type_);

	struct DeviceBuffer {
	private:
		void* data_ = nullptr;
		size_t size_ = 0;
		MemType mType;
	public:
		DeviceBuffer(size_t size, MemType memType = Device);

		template<typename T = void> T* get() { return (T*)data_; }
		template<typename T = void> T* data() { return (T*)data_; }
		size_t size(void) const { return size_; }

		~DeviceBuffer();

		DeviceBuffer(const DeviceBuffer& buf) = delete;
		DeviceBuffer& operator=(const DeviceBuffer& buf) = delete;
	};

	struct BufferManager {
	private:
		int devId;
		std::map<std::string, std::shared_ptr<DeviceBuffer>> buffers;
	public:
		BufferManager(void);

		int getDevice(void) { return devId; }

		template<typename T = char>
		std::shared_ptr<DeviceBuffer> addBuffer(const std::string& name, size_t arr_size, MemType mType = MemType::Device) {
			buffers[name] = std::make_shared<DeviceBuffer>(arr_size * sizeof(T), mType);
			return buffers[name];
		}

		std::string addBuffer(size_t arr_sze) {
			static size_t n_ano = 0;
			char buf[1000];
			sprintf_s(buf, "anonymous_%zu", n_ano++);
			printf("[buf] adding anonymous buffer %s...\n", buf);
			addBuffer(buf, arr_sze);
			return buf;
		}

		std::shared_ptr<DeviceBuffer> getBuffer(const std::string& name) {
			return buffers[name];
		}
		void deleteBuffer(const std::string& name) {
			buffers[name].reset();
		}

		bool deleteBuffer(void* pdata) {
			for (auto iter = buffers.begin(); iter != buffers.end(); iter++) {
				if (iter->second->get() == pdata) {
					buffers.erase(iter);
					return true;
				}
			}
			return false;
		}

		size_t size(void) const {
			size_t memsize = 0;
			for (auto iter = buffers.begin(); iter != buffers.end(); iter++) {
				memsize += iter->second->size();
			}
			return memsize;
		}

		size_t size(std::string namepattern) const;
	};

	BufferManager& getMem();
	void freeMem(void);
}

