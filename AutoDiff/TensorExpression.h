#pragma once
#include "AutoDiff.h"
#include <array>
#include <iterator>

#ifdef __CUDACC__
#include "culib/lib.cuh"
using namespace culib;
#endif

#pragma push_macro("max")
#pragma push_macro("min")
#undef max
#undef min

#define TENSOR_WITH_MATLAB

namespace homo {
	template<typename T, bool Const = false> struct TensorView;
#ifdef TENSOR_WITH_MATLAB
	extern void tensor2matlab(const std::string& tname, const TensorView<float>& tf);
	extern void tensor2matlab(const std::string& tname, const TensorView<double>& tf);
#endif
	extern void tensor2vdb(const std::string& fname, const TensorView<float>& tf);
	extern void tensor2vdb(const std::string& fname, const TensorView<double>& tf);
	extern void vdb2tensor(const std::string& fname, TensorView<float> tf, bool interpolation = true);
	extern void loadvdb(const std::string& filename, std::vector<int> pos[3], std::vector<float>& gridvalues);
	extern void tensorProject(TensorView<float> tf, float beta, float eta, float a, float b);

	struct HomoTraits;

	// ToDo : design densityExp based on vec_t
	// only support 1-3 dimension tensor
	template<typename T>
	struct TensorBuffer {
		cudaPitchedPtr ptr;
		cudaExtent dim;
		TensorBuffer(cudaExtent dim_) {
#ifdef __CUDACC__
			ptr.ptr = nullptr;
			dim = dim_;
			dim_.width *= sizeof(T);
			auto err = cudaMalloc3D(&ptr, dim_);
			if (err) {
				CheckErr(err);
				printf("memory allocation failed,  dim = (%zu, %zu, %zu), siz = %zu",
					dim_.width, dim_.height, dim_.depth, dim_.width * dim_.height * dim_.depth);
				throw std::runtime_error("failed to allocate 3d memory");
			}
#else
			dim = dim_;
			ptr.pitch = (dim_.width * sizeof(T) + 63) / 64 * 64;
			ptr.xsize = dim.width;
			ptr.ysize = dim.height * dim.depth;
			cudaMallocHost(&ptr.ptr, ptr.pitch * dim_.height * dim_.depth);
#endif
		}
		int getDim(int axis) const { 
			if (axis == 0) return dim.width;
			else if (axis == 1) return dim.height;
			else if (axis == 2) return dim.depth;
			else return -1;
		}
		void reset(T initval = 0) {
#ifdef __CUDACC__
			//cudaMemset(ptr.ptr, 0, total());
			init_array(data(), initval, size_elements());
#else
			for (int i = 0; i < size_elements(); i++) {
				((T*)(ptr.ptr))[i] = initval;
			}
#endif
		}
		size_t size(void) const {
			return ptr.pitch * dim.height * dim.depth;
		}
		size_t size_elements(void) const{
			return size() / sizeof(T);
		}
		size_t pitch(void) const { return ptr.pitch; }
		void copy(const TensorBuffer<T>& t2) {
			cudaMemcpy(ptr.ptr, t2.ptr.ptr, size(), cudaMemcpyDeviceToDevice);
		}
		~TensorBuffer() {
#ifdef __CUDACC__
			cudaFree(ptr.ptr);
#else
			cudaFreeHost(ptr.ptr);
#endif
		}
		T* data(void) {
			return (T*)ptr.ptr;
		}
	};

	template<typename T>
	struct TensorIterator
		: public std::iterator<std::random_access_iterator_tag, T>
	{
		size_t location, pitchT;
		int x;
		T* p_data;
		__host_device_func TensorIterator(T* pdata, int xT, size_t pitchT_) : pitchT(pitchT_), location(0), x(xT), p_data(pdata) {}
		__host_device_func __forceinline__ auto& operator[](size_t id) {
			id += location; return p_data[id % x + id / x * pitchT];
		}
		__host_device_func __forceinline__ const auto& operator[](size_t id) const {
			id += location; return p_data[id % x + id / x * pitchT];
		}
		__host_device_func __forceinline__ TensorIterator& operator+(size_t id) {
			TensorIterator iter(*this); iter.location += id;
			return iter;
		}
		__host_device_func __forceinline__ TensorIterator& operator-(size_t id) {
			TensorIterator iter(*this); iter.location -= id;
			return iter;
		}
		// ToDo : add other operations of random access iterator
	};

	template<typename T, bool ConstValue /*= false*/>
	struct TensorView {
		T* pdata;
		size_t pitchT;
		int dim[3];
		typedef T Scalar;

		TensorView(TensorBuffer<T>& buf) {
			pdata = reinterpret_cast<T*>(buf.ptr.ptr);
			pitchT = buf.ptr.pitch / sizeof(T);
			dim[0] = buf.dim.width;
			dim[1] = buf.dim.height;
			dim[2] = buf.dim.depth;
		}
		__host_device_func TensorView(const TensorView& view2) = default;
		__host_device_func size_t getPitchT(void) const { return pitchT; }
		__host_device_func auto begin(void) { return TensorIterator<T>(pdata, dim[0], pitchT); }
		template<typename T2> __host_device_func auto beginAs(void) {
			return TensorIterator<T2>(reinterpret_cast<T2*>(pdata), dim[0], pitchT);
		}
		__host_device_func T* data(void) { return pdata; }
		__host_device_func const T* data(void) const { return pdata; }
		__host_device_func int size(void) const{
			return dim[0] * dim[1] * dim[2];
		}
		__host_device_func int size(int i) const{
			return dim[i];
		}
		__host_device_func void index(int n, int id[3]) const {
			id[0] = n % dim[0];
			id[1] = n / dim[0] % dim[1];
			id[2] = n / dim[0] / dim[1];
		}
		__device_func T& operator()(int i) {
			int j = i / dim[0];
			return pdata[i % dim[0] + j * pitchT];
		}
		__device_func T& operator()(int i, int j) {
			return pdata[i + j * pitchT];
		}
		__device_func T& operator()(int i, int j, int k) {
			return pdata[i + (k * dim[1] + j) * pitchT];
		}

		__device_func const T& operator()(int i) const {
			int j = i / dim[0];
			return pdata[i % dim[0] + j * pitchT];
		}
		__device_func const T& operator()(int i, int j) const {
			return pdata[i + j * pitchT];
		}
		__device_func const T& operator()(int i, int j, int k) const {
			return pdata[i + (k * dim[1] + j) * pitchT];
		}

		DeviceAddressProxy<T> operator[](int i) {
			int j = i / dim[0] % dim[1];
			int k = i / (dim[0] * dim[1]);
			return DeviceAddressProxy<T>(pdata + i % dim[0] + (j + k * dim[1]) * pitchT);
		}

	};

	template<typename T>
	struct TensorView<T, true> {
		T constVal;
		size_t pitchT;
		int dim[3];
		typedef T Scalar;
		TensorView(TensorBuffer<T>& shapeLike, T val) : constVal(val) {
			pitchT = shapeLike.ptr.pitch / sizeof(T);
			dim[0] = shapeLike.dim.width;
			dim[1] = shapeLike.dim.height;
			dim[2] = shapeLike.dim.depth;
		}
		__host_device_func TensorView(const TensorView& view2) = default;
		__host_device_func int size(void) const{
			return dim[0] * dim[1] * dim[2];
		}
		__host_device_func int size(int i) const{
			return dim[i];
		}
		__host_device_func void index(int n, int id[3]) {
			id[0] = n % dim[0];
			id[1] = n / dim[0] % dim[1];
			id[2] = n / dim[0] / dim[1];
		}
		__device_func T operator()(int i) {
			return constVal;
		}
		__device_func T operator()(int i, int j) {
			return constVal;
		}
		__device_func T operator()(int i, int j, int k) {
			return constVal;
		}
	};

	enum  Order {
		i, j, k,
		ij, ik, jk,
		ji, ki, kj,
		ijk, ikj, jik, jki, kji, kij
	};

	enum TensorSym {
		Reflection3,
		Reflection6,
		Rotate3
	};

	template<typename T>
	struct Tensor 
	{
		std::shared_ptr<TensorBuffer<T>> buf;
		Tensor(int x, int y = 1, int z = 1)
			: buf(new TensorBuffer<T>(cudaExtent{ (size_t)x,(size_t)y,(size_t)z })) {
		}
		Tensor(std::array<int, 3> dims)
			: buf(new TensorBuffer<T>(cudaExtent{ (size_t)dims[0],(size_t)dims[1],(size_t)dims[2] })) {
		}
		Tensor(void) = default;
		operator bool(void) const { return bool(buf); }
		size_t size_elements(void) const { return buf->size_elements(); }
		void reset(T initval = 0) { buf->reset(initval); }
		void clear(void) { buf.reset(); }
		void rand(T low, T upp) {
#ifdef __CUDACC__
			tensor_rand(view(), low, upp);
#else
#endif
		}
#ifdef TENSOR_WITH_MATLAB
		void toMatlab(const std::string& tname) { tensor2matlab(tname, view()); }
#else
		void toMatlab(const std::string& tname) {}
#endif
		void toVdb(const std::string& fname) { tensor2vdb(fname, view()); }
		void fromVdb(const std::string& fname, bool interpolation = true) { vdb2tensor(fname, view(),interpolation); }
		void proj(float beta = 20.f, float eta = 0.5f, float a = 1.f, float b = 0.f) {
			tensorProject(view(), beta, eta, a, b);
		}
		template<int N >
		static Tensor<T> range(std::array<T, N> start, std::array<T, N> end, std::array<int, N> steps, Order axis) {
			if (axis >= 3) return Tensor<T>();
			//cudaExtent extents;
			//extents.width = steps[0] + 1;
			//if constexpr (N >= 1) extents.height = steps[1] + 1; else extents.height = 1;
			//if constexpr (N >= 2) extents.depth = steps[2] + 1; else extents.depth = 1;
			//steps[0] += 1; steps[1] += 1; steps[2] += 1;
			std::array<int, 3> newdims;
			for (int i = 0; i < 3; i++) if (i < N) newdims[i] = steps[i] + 1; else newdims[i] = 1;
			Tensor<T> rangeBuf(newdims);
			//buf = std::make_shared<TensorBuffer<T>>(extents);
#ifdef __CUDACC__
			devArray_t<T, 3> s1, s2;
			devArray_t<int, 3> ns;
			for (int i = 0; i < 3; i++) {
				if (i < N) {
					s1[i] = start[i]; s2[i] = end[i];
					ns[i] = steps[i];
				} else {
					s1[i] = 0; s2[i] = 0;
					ns[i] = 0;
				}
			}
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, rangeBuf.size_elements(), 256);
			range_kernel << <grid_size, block_size >> > (s1, s2, ns, rangeBuf.view(), axis);
			cudaDeviceSynchronize();
			cuda_error_check;
#else
			auto dataview = rangeBuf.view();
			T slen[3] = {
				(end[0] - start[0]) / steps[0],
				(end[1] - start[1]) / steps[1],
				(end[2] - start[2]) / steps[2] 
			};
			int siz = rangeBuf.size_elements();
			for (int i = 0; i < siz; i++) {
				int pos[3];
				dataview.index(i, pos);
				dataview(i) = pos[axis] * slen[axis] + start[axis];
			}
#endif
			return rangeBuf;
		}
		static Tensor<T> range(T start, T end, int steps) {
			std::array<T, 3> startlist{ start,0,0 };
			std::array<T, 3> endlist{ end,0,0 };
			std::array<int, 3> stepslist{ steps,0,0 };
			return range(startlist, endlist, stepslist, Order::i);
		}

		void copy(Tensor<T> t2) {
			buf->copy(*t2.buf);
		}
		std::array<int, 3> getDim(void) const {
			std::array<int, 3> di{ length(0),length(1),length(2) };
			//printf("di = %d, %d, %d\n", di[0], di[1], di[2]);
			return di;
		}
		int length(int axis) const {
			if (axis < 3) return  buf->getDim(axis);
		}
		int size(int k) const { return view().size(k); }
		int size(void) const { return view().size(); }
		T maxabs(void) {
#ifdef __CUDACC__
			auto iter = view().template beginAs<AbsWrapT<T>>();
			MaxOp<AbsWrapT<T>> maxop;
			T maxValue = sequence_reduce(iter, maxop, view().size(), AbsWrapT<T>(0));
			return maxValue;
#endif 
		}
		T max(void) {
#ifdef __CUDACC__
			auto iter = view().begin();
			MaxOp<T> maxop;
			T maxValue = sequence_reduce(iter, maxop, view().size(), std::numeric_limits<T>::lowest());
			return maxValue;
#endif 
		}
		T min(void) {
#ifdef __CUDACC__
			auto iter = view().begin();
			MinOp<T> minop;
			T minValue = sequence_reduce(iter, minop, view().size(), std::numeric_limits<T>::lowest());
			return minValue;
#endif 
		}
		T sum(void) {
#ifdef __CUDACC__
			auto iter = view().begin();
			SumOp<T> sumop;
			T sumValue = sequence_reduce(iter, sumop, view().size(), T{ 0 });
			return sumValue;
#endif 
		}
		Tensor<T> flatten(void) const {
			auto myview = view();
			Tensor<T> flat(myview.size());
			cudaMemcpy2D(flat.data(), myview.size(0) * sizeof(T), data(), pitch(),
				myview.size(0) * sizeof(T), myview.size(1) * myview.size(2), cudaMemcpyDeviceToDevice);
			return flat;
		}
		void graft(T* arr) {
			auto myview = view();
			cudaMemcpy2D(data(), pitch(),
				arr, myview.size(0) * sizeof(T),
				myview.size(0) * sizeof(T), myview.size(1) * myview.size(2), cudaMemcpyDeviceToDevice);
		}
#ifdef __CUDACC__
		template<typename InitFunc>
		void setValue(InitFunc func) {
			TensorView<T> myview = view();
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, myview.size(), 256);
			::map << <grid_size, block_size >> > (myview.size(), [=] __device__(int tid) {
				TensorView<T> vi = myview;
				int p[3] = {
					tid % vi.dim[0],
					tid / vi.dim[0] % vi.dim[1],
					tid / (vi.dim[0] * vi.dim[1])
				};
				vi(tid) = func(p[0], p[1], p[2]);
			});
			cudaDeviceSynchronize();
			cuda_error_check;
		}
#endif
		void symmetrize(TensorSym symtype, bool average = true) {
#ifdef __CUDACC__
			auto myview = view();
			int nrep;
			if (symtype == Reflection3) {
				nrep = (myview.size(0) / 2) * (myview.size(1) / 2) * (myview.size(2) / 2);
			}
			else if (symtype == Reflection6) {
				int res[3] = { myview.size(0) / 2, myview.size(1) / 2, myview.size(2) / 2 };
				nrep = (res[0] + 1) * (res[1] + 2) * (res[2] + 3) / 6;
			}
			else if (symtype == Rotate3) {
				int res[3] = { myview.size(0) / 2, myview.size(1) / 2, myview.size(2) / 2 };
				nrep = res[2] * (res[2] + 1) * (2 * res[2] + 1) / 6;
			}
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, nrep, 256);
			symetrize_tensor_kernel << <grid_size, block_size >> > (myview, symtype, average);
			cudaDeviceSynchronize();
			cuda_error_check;
#else
#endif
		}
		[[nodiscard]] T* data(void) { return buf->data(); }
		[[nodiscard]] const T* data(void) const { return buf->data(); }
		template<typename Lambda>
		void mapInplace(Lambda mapker) {
#ifdef __CUDACC__
			TensorView<T> myview = view();
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, myview.size(), 256);
			traverse_noret << <grid_size, block_size >> > (myview.size(), [=] __device__(int tid) {
				TensorView<T> vi = myview;
				int p[3];
				vi.index(tid, p);
				auto oldval = vi(tid);
				vi(tid) = mapker(p[0], p[1], p[2], oldval);
			});
			cudaDeviceSynchronize();
			cuda_error_check;
#else
#endif
		}
		size_t pitch(void) const { return buf->pitch(); }
		//size_t size(void) { return view().size(); }
		const TensorView<T> view(void) const {
			return TensorView<T>(*buf);
		}
		TensorView<T> view(void) {
			return TensorView<T>(*buf);
		}
	};


	template<typename Scalar, typename Kernel, typename opExp_t> struct unarymap_tsexp_t;
	template<typename Scalar, typename Kernel, typename opExp_t> struct conv_tsexp_t;
	template<typename Scalar, typename opExp> struct linear_umker_t;
	template<typename Scalar, typename opExp> struct pow_umker_t;
	template<typename Scalar, typename opExp> struct exp_umker_t;
	template<typename Scalar> struct eye_umker_t;

#define IS_TYPE_V(TypeName) \
template<typename Arg> struct is_##TypeName { static constexpr bool value = false; }; \
template<typename... Args> struct is_##TypeName< TypeName##_t<Args...> > { static constexpr bool value = true; }; \
template<typename Arg> constexpr bool is_##TypeName##_v = is_##TypeName<Arg>::value;

	IS_TYPE_V(linear_umker)
	IS_TYPE_V(pow_umker)
	IS_TYPE_V(unarymap_tsexp)
	IS_TYPE_V(exp_umker)

	template<typename subExp_t, typename T>
	struct tsexp_method_t {
		template<typename Scalar = T, 
			typename SubExp = subExp_t,
			std::enable_if_t<is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto pow(Scalar p) {
			pow_umker_t<T, eye_umker_t<T>> powker(eye_umker_t<T>(), p);
			auto& subexp = *static_cast<SubExp*>(this);
			return subexp.composite(powker);
		}

		template<typename Scalar = T, 
			typename SubExp = subExp_t,
			std::enable_if_t<!is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto pow(Scalar p) {
			using Kernel = pow_umker_t<T, eye_umker_t<T>>;
			Kernel powker(eye_umker_t<T>(), p);
			auto& subexp = *static_cast<SubExp*>(this);
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, powker);
		}

		template <typename Scalar = T,
				  typename SubExp = subExp_t,
				  std::enable_if_t<is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto exp(void) {
			exp_umker_t<T, eye_umker_t<T>> expker((eye_umker_t<T>()));
			auto &subexp = *static_cast<SubExp *>(this);
			return subexp.composite(expker);
		}

		template<typename Scalar = T, 
			typename SubExp = subExp_t,
			std::enable_if_t<!is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto exp(void) {
			using Kernel = exp_umker_t<T, eye_umker_t<T>>;
			Kernel expker((eye_umker_t<T>()));
			auto& subexp = *static_cast<SubExp*>(this);
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, expker);
		}

		template<typename Scalar = T, 
			typename SubExp = subExp_t,
			std::enable_if_t<is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto operator*(Scalar a) {
			linear_umker_t<T, eye_umker_t<T>> linker(eye_umker_t<T>(), a, 1);
			auto& subexp = *static_cast<SubExp*>(this);
			return  subexp.composite(linker);
		}

		template<typename Scalar = T, 
			typename SubExp = subExp_t,
			std::enable_if_t<!is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto operator*(Scalar a) {
			using Kernel = linear_umker_t<T, eye_umker_t<T>>;
			Kernel linker(eye_umker_t<T>(), a, 1);
			auto& subexp = *static_cast<SubExp*>(this);
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, linker);
		}

		template<typename Scalar = T, 
			typename SubExp = subExp_t,
			std::enable_if_t<is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto operator+(Scalar b) {
			linear_umker_t<T, eye_umker_t<T>> linker(eye_umker_t<T>(), 1, b);
			auto& subexp = *static_cast<SubExp*>(this);
			return subexp.composite(linker);
		}

		template<typename Scalar = T, 
			typename SubExp = subExp_t,
			std::enable_if_t<!is_unarymap_tsexp_v<SubExp>, int> = 0>
		__host_device_func auto operator+(Scalar b) {
			using Kernel = linear_umker_t<T, eye_umker_t<T>>;
			Kernel linker(eye_umker_t<T>(), 1, b);
			auto& subexp = *static_cast<SubExp*>(this);
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, linker);
		}

		template<typename ConvKernel>
		__host_device_func auto conv(ConvKernel kernel) {
			auto& subexp = *static_cast<subExp_t*>(this);
			return conv_tsexp_t <T, ConvKernel, subExp_t>(subexp, kernel);
		}

		__host_device_func auto operator/(T x) {
			return operator*(T{ 1. } / x);
		}

		__host_device_func auto operator-(T b) {
			return operator+(-b);
		}

	};

	template<typename subTensor, typename T = float>
	struct tsexp_t
		: public tsexp_method_t<subTensor, T>
	{
		Tensor<T> values;
		Tensor<T> diffs;
		Tensor<T> tempbuf;

		typedef T ScalarType;

		tsexp_t(Tensor<T> otherValues, Tensor<T> otherDiffs) : values(otherValues), diffs(otherDiffs) {}

		tsexp_t(Tensor<T> otherValues) :values(otherValues), diffs(otherValues.getDim()) {}

		tsexp_t(size_t xlen_, size_t ylen_ = 1, size_t zlen_ = 1)
			: values(xlen_, ylen_, zlen_), diffs(xlen_, ylen_, zlen_) { }

		//tsexp_t(void) : values(nullptr), diffs(nullptr), len(0) {}

		tsexp_t(std::array<int, 3> dims) :values(dims), diffs(dims) {}

		~tsexp_t() { }

		std::array<int, 3> getDim(void) const { return values.getDim(); }
		int length(int axis) { return values.length(axis); }
		//int n_total(void) { return values.size_elements(); }
		int n_valid(void) {
			auto d3 = values.getDim();
			return d3[0] * d3[1] * d3[2];
		}
		void requireTempbuf(void) {
			if (!tempbuf) {
				tempbuf = Tensor<T>(getDim());
			}
		}
		void releaseTempbuf(void) { tempbuf.clear(); }
		auto getTemp(void) { return tempbuf; }
		Tensor<T> value(void) const { return values; }
		Tensor<T> rvalue(void) { return values; }
		Tensor<T> diff(void) const { return diffs; }
		Tensor<T> rdiff(void) { return diffs; }
		Tensor<T> eval(void) {
			static_cast<subTensor*>(this)->eval_imp(values);
			diffs.reset();
			return values;
		}
		void backward(Tensor<T> lastdiff) {
			//diffs.copy(lastdiff);
			static_cast<subTensor*>(this)->backward_imp(lastdiff);
		}
	};
	
	template<typename Vec>
	struct scalar_traits {};

	template<typename subTensor, typename Scalar>
	struct scalar_traits<tsexp_t<subTensor, Scalar>> {
		using type = Scalar;
	};

	template<typename Scalar, typename subKer>
	struct um_ker_t {
		template<typename opKer, typename SubKer = subKer,
			std::enable_if_t<is_linear_umker_v<SubKer>, int> = 0
		> __host_device_func auto composite(const linear_umker_t<Scalar, opKer>& op) {
			auto& subker = static_cast<SubKer&>(*this);
			return linear_umker_t<Scalar, typename SubKer::opExp_t>(subker.op, subker.a * op.a, op.b + op.a * subker.b);
		}
		template<typename opKer, typename SubKer = subKer,
			std::enable_if_t<!is_linear_umker_v<SubKer>, int> = 0
		> __host_device_func auto composite(const linear_umker_t<Scalar, opKer>& op) {
			auto& subker = static_cast<SubKer&>(*this);
			return linear_umker_t<Scalar, SubKer>(subker, op.a, op.b);
		}
		template<typename opKer, typename SubKer = subKer,
			std::enable_if_t<is_pow_umker_v<SubKer>, int> = 0
		> __host_device_func auto composite(const pow_umker_t<Scalar, opKer>& op) {
			auto& subker = static_cast<SubKer&>(*this);
			return pow_umker_t<Scalar, typename SubKer::opExp_t>(subker.op, op.p + subker.p);
		}
		template<typename opKer, typename SubKer = subKer,
			std::enable_if_t<!is_pow_umker_v<SubKer>, int> = 0
		> __host_device_func auto composite(const pow_umker_t<Scalar, opKer>& op) {
			auto& subker = static_cast<SubKer&>(*this);
			return pow_umker_t<Scalar, SubKer>(subker, op.p);
		}
		template <typename opKer, typename SubKer = subKer,
				  std::enable_if_t<!is_exp_umker_v<SubKer>, int> = 0
		> __host_device_func auto composite(const exp_umker_t<Scalar, opKer> &op) {
			auto &subker = static_cast<SubKer &>(*this);
			return exp_umker_t<Scalar, SubKer>(subker);
		}
	};

	template<typename Scalar>
	struct eye_umker_t : 
		public um_ker_t<Scalar, eye_umker_t<Scalar>> 
	{
		template<typename rArg>
		__host_device_func auto evalExp(rArg arg) {
			return arg;
		}
	};


	template<typename Scalar, typename opExp>
	struct linear_umker_t : public um_ker_t<Scalar, linear_umker_t<Scalar, opExp>>
	{
		using opExp_t = opExp;
		// a_ * op + b_
		opExp op;
		using Base = um_ker_t<Scalar, linear_umker_t<Scalar, opExp>>;
		Scalar a = 1, b = 0;
		linear_umker_t(opExp op_, Scalar a_, Scalar b_) :op(op_), a(a_), b(b_) { }
		template<typename rArg>
		__host_device_func auto evalExp(rArg arg) {
			return op.template evalExp<rArg>(arg) * a + b;
		}
		//using Exp = decltype(((linear_umker_t*)nullptr)->evalExp(*((rvar_exp_t<Scalar>*)nullptr)));
		__host_device_func auto eval(var_exp_t<void,Scalar>& v) {
			auto expr = evalExp(v.ref());
			expr.eval();
			return expr;
		}
		__host_device_func Scalar eval(Scalar val, Scalar& diffe) {
			rvar_exp_t<Scalar> v(val, diffe);
			auto expr = evalExp(v);
			expr.eval();
			expr.backward(1);
			//diffe = expr.diff();
			return expr.value();
		}
	};

	template<typename Scalar, typename opExp>
	struct pow_umker_t : public um_ker_t<Scalar, pow_umker_t<Scalar, opExp>>
	{
		using opExp_t = opExp;
		opExp op;
		using Base = um_ker_t<Scalar, pow_umker_t<Scalar, opExp>>;
		Scalar p;
		pow_umker_t(opExp op_, Scalar pon) : op(op_), p(pon) {}

		template<typename rArg>
		__host_device_func auto evalExp(rArg arg) {
			return op.template evalExp<rArg>(arg).pow(p);
		}
		//using Exp = decltype(((pow_umker_t<Scalar, opExp>*)nullptr)->evalExp(*((rvar_exp_t<Scalar>*)nullptr)));
		__host_device_func auto eval(var_exp_t<void,Scalar>& v) {
			auto expr = evalExp(v.ref());
			expr.eval();
			expr.backward(1);
			return expr;
		}
		__host_device_func Scalar eval(Scalar val, Scalar& diffe) {
			rvar_exp_t<Scalar> v(val, diffe);
			auto expr = evalExp(v);
			expr.eval();
			expr.backward(1);
			//diffe = expr.diff();
			return expr.value();
		}
	};

	template<typename Scalar, typename opExp>
	struct exp_umker_t : public um_ker_t<Scalar, exp_umker_t<Scalar, opExp>>
	{
		using opExp_t = opExp;
		opExp op;
		using Base = um_ker_t<Scalar, exp_umker_t<Scalar, opExp>>;
		exp_umker_t(opExp op_) : op(op_) {}

		template<typename rArg>
		__host_device_func auto evalExp(rArg arg) {
			return op.template evalExp<rArg>(arg).exp();
		}
		//using Exp = decltype(((pow_umker_t<Scalar, opExp>*)nullptr)->evalExp(*((rvar_exp_t<Scalar>*)nullptr)));
		__host_device_func auto eval(var_exp_t<void,Scalar>& v) {
			auto expr = evalExp(v.ref());
			expr.eval();
			expr.backward(1);
			return expr;
		}
		__host_device_func Scalar eval(Scalar val, Scalar& diffe) {
			rvar_exp_t<Scalar> v(val, diffe);
			auto expr = evalExp(v);
			expr.eval();
			expr.backward(1);
			//diffe = expr.diff();
			return expr.value();
		}
	};

	template<typename Scalar, typename Kernel, typename opExp_t>
	struct unarymap_tsexp_t 
		:public  tsexp_t<unarymap_tsexp_t<Scalar, Kernel, opExp_t>, Scalar>
	{
		opExp_t op;
		Kernel ker;
		using Base = tsexp_t<unarymap_tsexp_t<Scalar, Kernel, opExp_t>, Scalar>;
		//using Scalar = typename scalar_traits<map_tsexp_t<Kernel, opExp_t>>::type;
		unarymap_tsexp_t(const opExp_t& op_, Kernel ker_)
			: op(op_), ker(ker_), Base(op_.getDim()) {}

		template<typename Ker2>
		__host_device_func auto composite(Ker2 ker2) {
			using NewKernel = decltype(ker.composite(ker2));
			return unarymap_tsexp_t<Scalar, NewKernel, opExp_t>(op, ker.composite(ker2));
		}


		__host_device_func void eval_imp(Tensor<Scalar> vals) {
			op.eval();
			Base::requireTempbuf();
#ifdef __CUDACC__
			size_t grid_size, block_size;
			auto accs = op.value().view();
			auto sdif = Base::getTemp().view();
			auto accd = Base::value().view();
			//op.value().toMatlab("umsrc");
			//cudaDeviceSynchronize();
			//cuda_error_check;
			make_kernel_param(&grid_size, &block_size, Base::n_valid(), 256);
			// (AccS accs, AccSdiff sdif, Accd accd, size_t n_tol, Kernel ker);
			unarymap_kernel << <grid_size, block_size >> > (accs, sdif, accd, Base::n_valid(), ker);
			cudaDeviceSynchronize();
			cuda_error_check;
			// DEBUG
			//Base::value().toMatlab("umvalue");
			//Base::getTemp().toMatlab("tmpdif");
#else
			auto valsview = vals.view();
			auto diffview = Base::getTemp().view();
			auto opview = op.value().view();
			for (int i = 0; i < Base::n_valid(); i++) {
				valsview(i) = ker.eval(opview(i), diffview(i));
			}
#endif
		}

		__host_device_func void backward_imp(Tensor<Scalar> lastdiff) {
			auto temp = Base::getTemp().view();
			auto lastdifview = lastdiff.view();
			auto src = op.value().view();
			auto srcDiff = op.diff().view();
#ifdef __CUDACC__
			//auto gradfunc = [=] __device__(typename opExp_t::Scalar s, Scalar dif) {
			//	auto kern = ker;
			//	return kern.grad(s) * dif;
			//};
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, Base::n_valid(), 256);
			//cudaDeviceSynchronize();
			//cuda_error_check;
			//(AccS accs, AccSdiff srcdif, TempDif tempdif, AccDdiff dstdif, size_t n_tol, Kernel ker)
			unarymap_backward << <grid_size, block_size >> > (src, srcDiff, temp, lastdifview, Base::n_valid(), ker);
			cudaDeviceSynchronize();
			cuda_error_check;
			//op.diff().toMatlab("umdiff");
#else
			for (int i = 0; i < Base::n_valid(); i++) {
				srcDiff(i) += lastdifview(i) * temp(i);
			}
#endif
			// TODO 
			//Base::releaseTempbuf();
			op.backward(op.diff());
		}
	};

	template<typename Scalar,typename WeightFunction>
	struct convker_t {
		Scalar pad = 0;
		bool period = false;
		int region[3];
		WeightFunction wfunc;
		Scalar wsum;
		bool normalize;
		__host_device_func convker_t(WeightFunction func, int halfdomain[3], Scalar padding /*= 0*/, bool normalize_ /*= false*/, bool period_/* = false*/)
			: pad(padding), wfunc(func), normalize(normalize_), period(period_) {
			region[0] = halfdomain[0];
			region[1] = halfdomain[1];
			region[2] = halfdomain[2];
			wsum = weightSum();
		}
		__host_device_func convker_t(WeightFunction func, int halfdomain, Scalar padding/* = 0*/, bool normalize_ /*= false*/, bool period_ /*= false*/) 
			: pad(padding), wfunc(func), normalize(normalize_), period(period_) {
			region[0] = halfdomain;
			region[1] = halfdomain;
			region[2] = halfdomain;
			wsum = weightSum();
		}
		__host_device_func int size(void) const {
			return (region[0] * 2 + 1) * (region[1] * 2 + 1) * (region[2] * 2 + 1);
		}
		__host_device_func Scalar weightSum(void) const {
			Scalar s = 0;
			int siz = size();
			for (int i = 0; i < siz; i++) {
				int off[3];
				neigh(i, off);
				s += wfunc(off);
			}
			return s;
		}
		__host_device_func bool is_period(void) const { return period; }
		__host_device_func Scalar padValue() const{ return pad; }
		__host_device_func void neigh(int id, int off[3]) const {
			off[0] = id % (region[0] * 2 + 1) - region[0];
			off[1] = id / (region[0] * 2 + 1) % (region[1] * 2 + 1) - region[1];
			off[2] = id / ((region[0] * 2 + 1) * (region[1] * 2 + 1)) - region[2];
		}
		__host_device_func Scalar weight(const int off[3]) const {
			Scalar w = wfunc(off);
			if (normalize) { w /= wsum; }
			//printf("wf = %f  wsum = %f\n", w, wsum);
			return w;
		}
	};

	template<typename WeightFunction, typename Scalar = float >
	convker_t<Scalar, WeightFunction> make_convker(WeightFunction func, int halfdomain[3], Scalar padding = 0, bool normalize = true, bool period = true) {
		return convker_t<Scalar, WeightFunction>(func, halfdomain, padding, normalize, period);
	}

	enum RadialConvWeight {
		Spline4,
		Linear
	};

	template<typename Scalar, RadialConvWeight WeightType> struct WeightFunc { };
	template<typename Scalar> struct WeightFunc<Scalar, Spline4> {
		Scalar r;
		__host_device_func WeightFunc<Scalar, Spline4>(Scalar r_) : r(r_) { }
		__host_device_func Scalar operator()(const int off[3]) const {
			Scalar r_off2 = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]) / (r * r);
			Scalar r_off = sqrt_(r_off2);
			Scalar val = 0;
			if (r_off2 < 1) val = 1 - 6 * r_off2 + 8 * r_off2 * r_off - 3 * r_off2 * r_off2;
			//printf("r2 = %f  r = %f\n", r_off2, r_off);
			return val;
		}
	};
	template<typename Scalar> struct WeightFunc<Scalar, Linear> {
		Scalar r;
		__host_device_func WeightFunc<Scalar, Linear>(Scalar r_) : r(r_) { }
		__host_device_func Scalar operator()(const int off[3]) const {
			Scalar r_off2 = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]) / (r * r);
			Scalar r_off = sqrt_(r_off2);
			Scalar val = 0;
			if (r_off2 < 1) val = 1 - r_off;
			//printf("r2 = %f  r = %f\n", r_off2, r_off);
			return val;
		}
	};

	template<typename Scalar, RadialConvWeight wType> struct radial_convker_t { };
	template<typename Scalar> struct radial_convker_t<Scalar, Spline4> 
		: convker_t<Scalar, WeightFunc<Scalar, Spline4>>
	{
		__host_device_func radial_convker_t<Scalar, Spline4>(Scalar r, Scalar padding = 0, bool normalize = true, bool period = false)
			: convker_t<Scalar, WeightFunc<Scalar, Spline4>>(
				WeightFunc<Scalar, Spline4>(r), r + 0.5, padding, normalize, period) {}
	};
	template<typename Scalar> struct radial_convker_t<Scalar, Linear> 
		: convker_t<Scalar, WeightFunc<Scalar, Linear>>
	{
		__host_device_func radial_convker_t<Scalar, Linear>(Scalar r, Scalar padding = 0, bool normalize = true, bool period = false)
			: convker_t<Scalar, WeightFunc<Scalar, Linear>>(
				WeightFunc<Scalar, Linear>(r), r + 0.5, padding, normalize, period) {}
	};

	template<typename Scalar, typename Kernel, typename opExp_t>
	struct conv_tsexp_t
		: public tsexp_t<conv_tsexp_t<Scalar, Kernel, opExp_t>, Scalar>
	{
		Kernel ker;
		opExp_t op;

		using Base = tsexp_t<conv_tsexp_t<Scalar, Kernel, opExp_t>, Scalar>;

		conv_tsexp_t(const opExp_t& op_, Kernel kernel) 
			: ker(kernel), op(op_), Base(op_.getDim()) {}

		__host_device_func void eval_imp(Tensor<Scalar> vals) {
			op.eval();
			Base::requireTempbuf();
			auto srcAcc = op.value().view();
			auto dstAcc = vals.view();
			auto tmpacc = Base::getTemp().view();
#ifdef __CUDACC__
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, Base::n_valid(), 256);
			conv_kernel << <grid_size, block_size >> > (Base::n_valid(), srcAcc, dstAcc, tmpacc, ker);
			cudaDeviceSynchronize();
			cuda_error_check;
			// DEBUG
			//vals.toMatlab("srvalue");
			//Base::value().toMatlab("convvalue");
#else
			typename opExp_t::ScalarType padding = ker.padValue();
			for (int k = 0; k < dstAcc.dim[2]; k++) {
				for (int j = 0; j < dstAcc.dim[1]; j++) {
					for (int i = 0; i < dstAcc.dim[0]; i++) {
						Scalar sum(0);
						for (int neighid = 0; neighid < ker.size(); neighid++) {
							int off[3];
							ker.neigh(neighid, off);
							Scalar w = ker.weight(neighid);
							typename opExp_t::ScalarType val = padding;
							int neighpos[3]{ off[0] + i, off[1] + j, off[2] + k };
							if (neighpos[0] >= 0 && neighpos[0] < srcAcc.dim[0] &&
								neighpos[1] >= 0 && neighpos[1] < srcAcc.dim[1] &&
								neighpos[2] >= 0 && neighpos[2] < srcAcc.dim[2]) {
								val = srcAcc(neighpos[0], neighpos[1], neighpos[2]);
							} 	
							sum += w * val;
						}
						dstAcc(i, j, k) = sum;
					}
				}
			}
#endif
		}

		__host_device_func void backward_imp(Tensor<Scalar> lastdiff) {
			auto dstAcc = lastdiff.view();
			auto srcAcc = op.diff().view();
			auto tmpacc = Base::getTemp().view();
#ifdef __CUDACC__
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, Base::n_valid(), 256);
			conv_backward_kernel << <grid_size, block_size >> > (Base::n_valid(), srcAcc, dstAcc, tmpacc, ker);
			cudaDeviceSynchronize();
			cuda_error_check;
			//lastdiff.toMatlab("srcdiff");
			//op.diff().toMatlab("convdiff");
#else
			for (int k = 0; k < srcAcc.dim[2]; k++) {
				for (int j = 0; j < srcAcc.dim[1]; j++) {
					for (int i = 0; i < srcAcc.dim[0]; i++) {
						Scalar sum(0);
						for (int neighid = 0; neighid < ker.size(); neighid++) {
							int off[3];
							ker.neigh(neighid, off);
							off[0] = i - off[0]; off[1] = j - off[1]; off[2] = k - off[2];
							Scalar w = ker.weight(neighid);
							typename decltype(dstAcc)::Scalar val{ 0 };
							if (off[0] >= 0 && off[0] < dstAcc.dim[0] &&
								off[1] >= 0 && off[1] < dstAcc.dim[1] &&
								off[2] >= 0 && off[2] < dstAcc.dim[2]) {
								val = dstAcc(off[0], off[1], off[2]);
							}
							sum += w * val;
						}
						srcAcc(i, j, k) = sum;
					}
				}
			}
#endif
			op.backward(op.diff());
		}
	};
	// define full expression
	template<typename Scalar>
	struct AssociativeOp {
		Scalar result = 0;
		Scalar result_grad = 0;
		__host_device_func void reset(void) {
			result = 0;
			result_grad = 0; 
		}
	};

	template<typename Op>
	struct is_associative { static constexpr bool value = false; };

	template<typename Scalar, template<typename, typename...> class Op>
	struct is_associative<Op<Scalar>> {
		static constexpr bool value = std::is_convertible_v<Op<Scalar>, AssociativeOp<Scalar>>;
	};

	template<typename Op> constexpr bool is_associative_v = is_associative<Op>::value;

	template<typename Scalar,typename ScalarConstant>
	struct PnormOp : public AssociativeOp<Scalar>
	{
		using Base = AssociativeOp<Scalar>;
		constexpr static Scalar p_ = ScalarConstant::value;
		__host_device_func void accept(Scalar v) {
			Base::result += pow_<Scalar>(v, p_);
		}
		__host_device_func Scalar accepted(void) {
			return Base::result;
		}
		__host_device_func Scalar reduce(void) {
			Scalar res = pow_<Scalar>(Base::result, 1. / p_);
			Base::result_grad = Scalar{ 1 } / p_ * pow_<Scalar>(Base::result, 1. / p_ - 1);
			return res;
		}
		__host_device_func Scalar reduce(Scalar sum) {
			Base::result = sum;
			Scalar res = pow_<Scalar>(Base::result, 1. / p_);
			Base::result_grad = Scalar{ 1 } / p_ * pow_<Scalar>(Base::result, 1. / p_ - 1);
			return res;
		}
		__host_device_func Scalar grad(Scalar v) {
			return p_ * pow_<Scalar>(v, p_ - 1) * Base::result_grad;
		}
		__host_device_func Scalar grad_result(Scalar res) {
			return Scalar(1) / p_ * pow_<Scalar>(res, 1 - p_);
		}
		__host_device_func Scalar grad(Scalar v, Scalar resultGrad) {
			return p_ * pow_<Scalar>(v, p_ - 1) * resultGrad;
		}
	};

	template<typename Scalar, typename ScalarConstant>
	struct LogSumExpOp : public AssociativeOp<Scalar>
	{
		using Base = AssociativeOp<Scalar>;
		constexpr static Scalar p_ = ScalarConstant::value;
		__host_device_func void accept(Scalar v) {
			Base::result += exp_<Scalar>(v * p_);
		}
		__host_device_func Scalar accepted(void) {
			return Base::result;
		}
		__host_device_func Scalar reduce(void) {
			Scalar res = log_<Scalar>(Base::result) / p_;
			Base::result_grad = 1. / Base::result;
			return res;
		}
		__host_device_func Scalar reduce(Scalar sum) {
			Base::result = sum;
			Scalar res = log_<Scalar>(Base::result) / p_;
			Base::result_grad = 1. / Base::result;
			return res;
			
		}
		__host_device_func Scalar grad(Scalar v) {
			return exp_<Scalar>(v*p_) * Base::result_grad;
		}
		__host_device_func Scalar grad_result(Scalar result) {
			return Scalar(1) / exp_<Scalar>(result * p_);
		}
		__host_device_func Scalar grad(Scalar v, Scalar resultGrad) {
			return exp_<Scalar>(v * p_) * resultGrad;
		}
	};

	template<typename Scalar, typename Kernel, typename opExp_t,
		typename ReduceOp, std::enable_if_t<is_associative_v<ReduceOp>, int > = 0>
	struct full_tsexp_t 
		:public tsexp_t<full_tsexp_t<Scalar, Kernel, opExp_t, ReduceOp>, Scalar>
	{
		using Base = tsexp_t<full_tsexp_t<Scalar, Kernel, opExp_t, ReduceOp>>;
		Kernel ker;
		opExp_t op;
		Tensor<ReduceOp> reduceRes;
		// group information
		//ForwardConfig forwardconf;
		//BackwardConfig backwardconf;

		// ** ** ** template Configs ** ** ** **  
		struct DefaultForwardConfig {
			// for stage
			static constexpr int szSrcBatch = 4;
			static constexpr int blockSize = 512;

		};
		struct DefaultBackwardConfig {
			// for stage
			static constexpr int szSrcBatch = 4;
			static constexpr int n_dstpass = 4;
			static constexpr int n_srcpass = 4;
			static constexpr int blockSize = 512;

			// for gather
			 static constexpr int BlockSizeGather = 256;
			 static constexpr int sBatchGather = 4;
			 static constexpr int n_parampass = 2;
		};
		// ** ** ** ** ** ** ** ** ** ** ** ** ** ** ** **  

		__host_device_func full_tsexp_t(Kernel kerfunc, const opExp_t& opExp)
			: ker(kerfunc), op(opExp), reduceRes(op.getDim()) { }

		__host_device_func void eval_imp(Tensor<Scalar> vals) {
			DefaultForwardConfig conf;
			Base::op.eval();
			auto srcAcc = Base::op.value().view();
			auto dstAcc = Base::vals.view();
#ifdef __CUDACC__
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, Base::n_valid(), conf.blockSize());
			fullCon_Reduce << <grid_size, block_size >> > (conf, srcAcc, dstAcc, ker, reduceRes.view());
			cudaDeviceSynchronize();
			cuda_error_check;
#else

#endif
		}

		__host_device_func void backward_imp(Tensor<Scalar> lastdiff) {
			using BackwardConfig = DefaultBackwardConfig;
			BackwardConfig conf;
			auto srcAcc = Base::op.value().view();
			auto dstAcc = Base::vals.view();
			auto gradsAcc = Base::op.diff().view();
#ifdef __CUDACC__
			using Ts = typename opExp_t::ScalarType;
			int n_dststride = ceilDiv(dstAcc.size(), BackwardConfig::n_dstpass);
			int nBlocksInDststride = ceilDiv(n_dststride, BackwardConfig::blockSize);
			int n_nBlocksInDststride = ceilDiv(srcAcc.size(), BackwardConfig::szSrcBatch * BackwardConfig::n_srcpass);
			Tensor<Ts> gradstemp(srcAcc.size(), nBlocksInDststride);
			size_t grid_size, block_size;
			make_kernel_param(&grid_size, &block_size, n_nBlocksInDststride, 256);
			fullCon_Reduce_backward_stage << <grid_size, block_size >> > (conf, srcAcc, dstAcc, ker, reduceRes.view(), gradstemp.view());
			cudaDeviceSynchronize();

			int sizeGroupGrads = gradstemp.size(1);
			int n_gradpass = sizeGroupGrads / BackwardConfig::BlockSizeGather;
			int nBlocksParam = ceilDiv(srcAcc.size() / Kernel::szParamGroup, BackwardConfig::n_parampass * BackwardConfig::sBatchGather);
			make_kernel_param(&grid_size, &block_size, nBlocksParam* BackwardConfig::BlockSizeGather, BackwardConfig::BlockSizeGather);
			fullCon_Reduce_backward << <grid_size, block_size >> > (conf, srcAcc, ker, gradstemp.view(), gradsAcc);
			cudaDeviceSynchronize();
#else
#endif
			op.backward(op.diff());
		}
	};

	template<typename Scalar, typename Kernel, typename opExp_t>
	struct sparse_tsexp_t
		: public tsexp_t<sparse_tsexp_t<Scalar, Kernel, opExp_t>, Scalar>
	{
		
	};

	template<typename subVar = void, typename Scalar = float>
	struct var_tsexp_t
		: tsexp_t<var_tsexp_t<subVar, Scalar>, Scalar>
	{
		//Tensor<Scalar>& vars;
		//Tensor<Scalar>& vardiffs;
		using Base = tsexp_t<var_tsexp_t<subVar, Scalar>, Scalar>;
		//var_tsexp_t(int xlen, int ylen, int zlen) :vars(Base::values), vardiffs(Base::diffs), Base(xlen, ylen, zlen) { vardiffs.reset(); }
		__host_device_func var_tsexp_t(int xlen, int ylen, int zlen) : Base(xlen, ylen, zlen) { Base::diff().reset(); }
		__host_device_func var_tsexp_t(Tensor<Scalar> val, Tensor<Scalar> dif) : Base(val, dif) {}
		__host_device_func var_tsexp_t(Tensor<Scalar> val) : Base(val) {}
		__host_device_func void eval_imp(Tensor<Scalar> vals) {
			// ToDo : Do nothing
			return;
		}
		__host_device_func void backward_imp(Tensor<Scalar> lastdiff) {
			// ToDo : Do nothing
			return;
		}
	};

	// define type alias
	template<typename T = float> using TensorVar = var_tsexp_t<void, T>;
}


#pragma pop_macro("max")
#pragma pop_macro("min")
