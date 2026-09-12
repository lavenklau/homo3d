#pragma once
#include "AutoDiff.h"
#include <array>
#include <iterator>
#include <tuple>

#ifdef __CUDACC__
#include "culib/lib.cuh"
using namespace culib;
// #include "TensorExpression.cuh"
#endif

#pragma push_macro("max")
#pragma push_macro("min")
#undef max
#undef min

namespace homo {
template<typename T, bool Const = false>
struct TensorView;
extern void tensor2vdb(const std::string& fname, const TensorView<float>& tf);
extern void tensor2vdb(const std::string& fname, const TensorView<double>& tf);
extern void vdb2tensor(const std::string& fname, TensorView<float> tf, bool interpolation = true);
extern void loadvdb(const std::string& filename, std::vector<int> pos[3], std::vector<float>& gridvalues);
extern void tensorProject(TensorView<float> tf, float beta, float eta, float a, float b);

template<int BlockSize, bool omitPreMap, typename AccS, typename Accd, typename ReduceOp,
		 std::enable_if_t<(BlockSize == 512), int> = 0>
__global__ void tensor_reduce(size_t n_src, AccS accs, Accd accd, ReduceOp op);
template<int BlockSize, typename Scalar, typename AccS, typename Accd, typename ReduceOp,
		 std::enable_if_t<(BlockSize == 512), int> = 0>
__global__ void tensor_reduce_gradient(AccS accs, Accd grad, Scalar res, Scalar lastdiff, ReduceOp op);

struct HomoTraits;

// ToDo : design densityExp based on vec_t
// only support 1-3 dimension tensor
template<typename T>
struct TensorBuffer {
	cudaPitchedPtr ptr;
	cudaExtent dim;
	TensorBuffer(cudaExtent dim_) {
		ptr.ptr = nullptr;
		dim = dim_;
		dim_.width *= sizeof(T);
		ptr.pitch = (dim_.width + 511) / 512 * 512;
		ptr.xsize = dim_.width / sizeof(T);
		ptr.ysize = dim_.depth * dim_.height;
		auto err = cudaMallocManaged(&ptr.ptr, ptr.pitch * ptr.ysize);
		if (err) {
			CheckErr(err);
			printf("memory allocation failed,  dim = (%zu, %zu, %zu), siz = %zu",
				   dim_.width, dim_.height, dim_.depth, dim_.width * dim_.height * dim_.depth);
			throw std::runtime_error("failed to allocate 3d memory");
		}
	}
	int getDim(int axis) const {
		if (axis == 0)
			return dim.width;
		else if (axis == 1)
			return dim.height;
		else if (axis == 2)
			return dim.depth;
		else
			return -1;
	}
	void reset(T initval = 0) {
		init_array(data(), initval, size_elements());
	}
	size_t size(void) const {
		return ptr.pitch * dim.height * dim.depth;
	}
	size_t size_elements(void) const {
		return size() / sizeof(T);
	}
	size_t pitch(void) const {
		return ptr.pitch;
	}
	void copy(const TensorBuffer<T>& t2) {
		cudaMemcpy(ptr.ptr, t2.ptr.ptr, size(), cudaMemcpyDeviceToDevice);
	}
	~TensorBuffer() {
		cudaFree(ptr.ptr);
	}
	T* data(void) {
		return (T*)ptr.ptr;
	}
};

template<typename T>
struct TensorIterator
	: public std::iterator<std::random_access_iterator_tag, T> {
	size_t location, pitchT;
	int x;
	T* p_data;
	__host_device_func TensorIterator(T* pdata, int xT, size_t pitchT_)
		: pitchT(pitchT_), location(0), x(xT), p_data(pdata) {}
	__host_device_func __forceinline__ auto& operator[](size_t id) {
		id += location;
		return p_data[id % x + id / x * pitchT];
	}
	__host_device_func __forceinline__ const auto& operator[](size_t id) const {
		id += location;
		return p_data[id % x + id / x * pitchT];
	}
	__host_device_func __forceinline__ TensorIterator operator+(size_t id) const {
		TensorIterator iter(*this);
		iter.location += id;
		return iter;
	}
	__host_device_func __forceinline__ TensorIterator operator-(size_t id) const {
		TensorIterator iter(*this);
		iter.location -= id;
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
	__host_device_func size_t getPitchT(void) const {
		return pitchT;
	}
	__host_device_func auto begin(void) {
		return TensorIterator<T>(pdata, dim[0], pitchT);
	}
	template<typename T2>
	__host_device_func auto beginAs(void) {
		return TensorIterator<T2>(reinterpret_cast<T2*>(pdata), dim[0], pitchT);
	}
	__host_device_func T* data(void) {
		return pdata;
	}
	__host_device_func const T* data(void) const {
		return pdata;
	}
	__host_device_func int size(void) const {
		return dim[0] * dim[1] * dim[2];
	}
	__host_device_func int size(int i) const {
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

enum Order {
	i,
	j,
	k,
	ij,
	ik,
	jk,
	ji,
	ki,
	kj,
	ijk,
	ikj,
	jik,
	jki,
	kji,
	kij
};

enum TensorSym {
	Reflection3,
	Reflection6,
	Rotate3
};

template<typename T>
struct Tensor {
	std::shared_ptr<TensorBuffer<T>> buf;
	Tensor(int x, int y = 1, int z = 1)
		: buf(new TensorBuffer<T>(cudaExtent{(size_t)x, (size_t)y, (size_t)z})) {
	}
	Tensor(std::array<int, 3> dims)
		: buf(new TensorBuffer<T>(cudaExtent{(size_t)dims[0], (size_t)dims[1], (size_t)dims[2]})) {
	}
	Tensor(void) = default;
	operator bool(void) const {
		return bool(buf);
	}
	size_t size_elements(void) const {
		return buf->size_elements();
	}
	void reset(T initval = 0) {
		buf->reset(initval);
	}
	void clear(void) {
		buf.reset();
	}
	void rand(T low, T upp) {
		tensor_rand(view(), low, upp);
	}
	void toVdb(const std::string& fname) {
		tensor2vdb(fname, view());
	}
	void fromVdb(const std::string& fname, bool interpolation = true) {
		vdb2tensor(fname, view(), interpolation);
	}
	void proj(float beta = 20.f, float eta = 0.5f, float a = 1.f, float b = 0.f) {
		tensorProject(view(), beta, eta, a, b);
	}
	void copy(Tensor<T> t2) {
		buf->copy(*t2.buf);
	}
	std::array<int, 3> getDim(void) const {
		std::array<int, 3> di{length(0), length(1), length(2)};
		//printf("di = %d, %d, %d\n", di[0], di[1], di[2]);
		return di;
	}
	int length(int axis) const {
		if (axis < 3)
			return buf->getDim(axis);
		return 0;
	}
	int size(int k) const {
		return view().size(k);
	}
	int size(void) const {
		return view().size();
	}
	T maxabs(void) {
#ifdef __CUDACC__
		auto iter = view().template beginAs<AbsWrapT<T>>();
		MaxOp<AbsWrapT<T>> maxop;
		T maxValue = sequence_reduce(iter, maxop, view().size(), AbsWrapT<T>(0));
		return maxValue;
#endif
	}
	T Sum(void) {
#ifdef __CUDACC__
		auto iter = view().begin();
		SumOp<T> sumop;
		T sumValue = sequence_reduce(iter, sumop, view().size(), T{0});
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
		auto cfg = make_kernel_param(myview.size(), 256);
		::map<<<cfg.grid, cfg.block>>>(myview.size(), [=] __device__(int tid) {
			TensorView<T> vi = myview;
			int p[3] = {
				tid % vi.dim[0],
				tid / vi.dim[0] % vi.dim[1],
				tid / (vi.dim[0] * vi.dim[1])};
			vi(tid) = func(p[0], p[1], p[2]);
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}
#endif
	void symmetrize(TensorSym symtype, bool average = true) {
		auto myview = view();
		int nrep;
		if (symtype == Reflection3) {
			nrep = (myview.size(0) / 2) * (myview.size(1) / 2) * (myview.size(2) / 2);
		} else if (symtype == Reflection6) {
			int res[3] = {myview.size(0) / 2, myview.size(1) / 2, myview.size(2) / 2};
			nrep = (res[0] + 1) * (res[1] + 2) * (res[2] + 3) / 6;
		} else if (symtype == Rotate3) {
			int res[3] = {myview.size(0) / 2, myview.size(1) / 2, myview.size(2) / 2};
			nrep = res[2] * (res[2] + 1) * (2 * res[2] + 1) / 6;
		}
		auto cfg = make_kernel_param(nrep, 256);
		symetrize_tensor_kernel<<<cfg.grid, cfg.block>>>(myview, symtype, average);
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	[[nodiscard]] T* data(void) {
		return buf->data();
	}
	[[nodiscard]] const T* data(void) const {
		return buf->data();
	}
	template<typename Lambda>
	void mapInplace(Lambda mapker) {
		TensorView<T> myview = view();
		auto cfg = make_kernel_param(myview.size(), 256);
		traverse_noret<<<cfg.grid, cfg.block>>>(myview.size(), [=] __device__(int tid) {
			TensorView<T> vi = myview;
			int p[3];
			vi.index(tid, p);
			auto oldval = vi(tid);
			vi(tid) = mapker(p[0], p[1], p[2], oldval);
		});
		cudaDeviceSynchronize();
		cuda_error_check;
	}
	void clamp(T low, T upp) {
		mapInplace([=] __device__(int i, int j, int k, T val) {
				if(val < low) return low;
				else if(val > upp) return upp;
				else if(low <= val && val <=upp) { 
					return val;
				}
				 else {
					return low;
				} });
	}
	size_t pitch(void) const {
		return buf->pitch();
	}
	//size_t size(void) { return view().size(); }
	const TensorView<T> view(void) const {
		return TensorView<T>(*buf);
	}
	TensorView<T> view(void) {
		return TensorView<T>(*buf);
	}
};

template<typename Scalar, typename Kernel, typename opExp_t>
struct unarymap_tsexp_t;
template<typename Scalar, typename Kernel, typename opExp_t>
struct conv_tsexp_t;
template<typename Scalar, typename opExp>
struct linear_umker_t;
template<typename Scalar, typename opExp>
struct pow_umker_t;
template<typename Scalar, typename opExp>
struct exp_umker_t;
template<typename Scalar, typename opExp>
struct sigmoid_umker_t;
template<typename Scalar>
struct eye_umker_t;
template<typename Scalar, typename Operand, typename ReduceOp>
struct reduce_tsexp_t;
template<typename Scalar>
struct SumReduceOp;
template<typename Scalar>
struct LogSumExpOp;
template<typename Scalar>
struct PnormReduceOp;

#define IS_TYPE_V(TypeName)                                                              \
	template<typename Arg>                                                               \
	struct is_##TypeName { static constexpr bool value = false; };                       \
	template<typename... Args>                                                           \
	struct is_##TypeName<TypeName##_t<Args...>> { static constexpr bool value = true; }; \
	template<typename Arg>                                                               \
	constexpr bool is_##TypeName##_v = is_##TypeName<Arg>::value;

IS_TYPE_V(linear_umker)
IS_TYPE_V(pow_umker)
IS_TYPE_V(unarymap_tsexp)
IS_TYPE_V(exp_umker)

template<typename subExp_t, typename T>
struct tsexp_method_t {
	template<typename Scalar = T, typename SubExp = subExp_t>
	__host_device_func auto pow(Scalar p) {
		using Kernel = pow_umker_t<T, eye_umker_t<T>>;
		Kernel powker(eye_umker_t<T>(), p);
		auto& subexp = *static_cast<SubExp*>(this);
		if constexpr (is_unarymap_tsexp_v<SubExp>) {
			return subexp.composite(powker);
		} else {
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, powker);
		}
	}

	template<typename Scalar = T, typename SubExp = subExp_t>
	__host_device_func auto exp(void) {
		using Kernel = exp_umker_t<T, eye_umker_t<T>>;
		Kernel expker((eye_umker_t<T>()));
		auto& subexp = *static_cast<SubExp*>(this);
		if constexpr (is_unarymap_tsexp_v<SubExp>) {
			return subexp.composite(expker);
		} else {
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, expker);
		}
	}

	template<typename Scalar = T, typename SubExp = subExp_t>
	__host_device_func auto sgm(Scalar k = 30, Scalar c = 0.5) {
		using Kernel = sigmoid_umker_t<T, eye_umker_t<T>>;
		Kernel sgmker((eye_umker_t<T>()), k, c);
		auto& subexp = *static_cast<SubExp*>(this);
		if constexpr (is_unarymap_tsexp_v<SubExp>) {
			return subexp.composite(sgmker);
		} else {
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, sgmker);
		}
	}

	template<typename Scalar = T, typename SubExp = subExp_t>
	__host_device_func auto operator*(Scalar a) {
		using Kernel = linear_umker_t<T, eye_umker_t<T>>;
		Kernel linker(eye_umker_t<T>(), a, 1);
		auto& subexp = *static_cast<SubExp*>(this);
		if constexpr (is_unarymap_tsexp_v<SubExp>) {
			return subexp.composite(linker);
		} else {
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, linker);
		}
	}

	template<typename Scalar = T, typename SubExp = subExp_t>
	__host_device_func auto operator+(Scalar b) {
		using Kernel = linear_umker_t<T, eye_umker_t<T>>;
		Kernel linker(eye_umker_t<T>(), 1, b);
		auto& subexp = *static_cast<SubExp*>(this);
		if constexpr (is_unarymap_tsexp_v<SubExp>) {
			return subexp.composite(linker);
		} else {
			return unarymap_tsexp_t<T, Kernel, SubExp>(subexp, linker);
		}
	}

	template<typename ConvKernel>
	__host_device_func auto conv(ConvKernel kernel) {
		auto& subexp = *static_cast<subExp_t*>(this);
		return conv_tsexp_t<T, ConvKernel, subExp_t>(subexp, kernel);
	}

	__host_device_func auto sum(void) {
		auto& subexp = *static_cast<subExp_t*>(this);
		return reduce_tsexp_t<T, subExp_t, SumReduceOp<T>>(subexp, SumReduceOp<T>());
	}

	__host_device_func auto pnorm(T p) {
		auto& subexp = *static_cast<subExp_t*>(this);
		return reduce_tsexp_t<T, subExp_t, PnormReduceOp<T>>(subexp, PnormReduceOp<T>(p));
	}

	__host_device_func auto logsumexp(void) {
		auto& subexp = *static_cast<subExp_t*>(this);
		return reduce_tsexp_t<T, subExp_t, LogSumExpOp<T>>(subexp, LogSumExpOp<T>());
	}

	__host_device_func auto operator/(T x) {
		return operator*(T{1.} / x);
	}

	__host_device_func auto operator-(T b) {
		return operator+(-b);
	}

	//__host_device_func auto Sum()
};

template<typename subTensor, typename T = float>
struct tsexp_t
	: public tsexp_method_t<subTensor, T> {
	Tensor<T> values;
	Tensor<T> diffs;
	Tensor<T> tempbuf;

	typedef T ScalarType;

	tsexp_t(Tensor<T> otherValues, Tensor<T> otherDiffs)
		: values(otherValues), diffs(otherDiffs) {}

	tsexp_t(Tensor<T> otherValues)
		: values(otherValues), diffs(otherValues.getDim()) {}

	tsexp_t(size_t xlen_, size_t ylen_ = 1, size_t zlen_ = 1)
		: values(xlen_, ylen_, zlen_), diffs(xlen_, ylen_, zlen_) {}

	//tsexp_t(void) : values(nullptr), diffs(nullptr), len(0) {}

	tsexp_t(std::array<int, 3> dims)
		: values(dims), diffs(dims) {}

	~tsexp_t() {}

	std::array<int, 3> getDim(void) const {
		return values.getDim();
	}
	int length(int axis) {
		return values.length(axis);
	}
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
	void releaseTempbuf(void) {
		tempbuf.clear();
	}
	auto getTemp(void) {
		return tempbuf;
	}
	Tensor<T> value(void) const {
		return values;
	}
	Tensor<T> rvalue(void) {
		return values;
	}
	Tensor<T> diff(void) const {
		return diffs;
	}
	Tensor<T> rdiff(void) {
		return diffs;
	}
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
	template<typename opKer, typename SubKer = subKer>
	__host_device_func auto composite(const linear_umker_t<Scalar, opKer>& op) {
		auto& subker = static_cast<SubKer&>(*this);
		if constexpr (is_linear_umker_v<SubKer>) {
			return linear_umker_t<Scalar, typename SubKer::opExp_t>(subker.op, subker.a * op.a, op.b + op.a * subker.b);
		} else {
			return linear_umker_t<Scalar, SubKer>(subker, op.a, op.b);
		}
	}
	template<typename opKer, typename SubKer = subKer>
	__host_device_func auto composite(const pow_umker_t<Scalar, opKer>& op) {
		auto& subker = static_cast<SubKer&>(*this);
		if constexpr (is_pow_umker_v<SubKer>) {
			return pow_umker_t<Scalar, typename SubKer::opExp_t>(subker.op, op.p + subker.p);
		} else {
			return pow_umker_t<Scalar, SubKer>(subker, op.p);
		}
	}
	template<typename opKer, typename SubKer = subKer>
	__host_device_func auto composite(const exp_umker_t<Scalar, opKer>& op) {
		auto& subker = static_cast<SubKer&>(*this);
		return exp_umker_t<Scalar, SubKer>(subker);
	}
	template<typename opKer, typename SubKer = subKer>
	__host_device_func auto composite(const sigmoid_umker_t<Scalar, opKer>& op) {
		auto& subker = static_cast<SubKer&>(*this);
		return sigmoid_umker_t<Scalar, SubKer>(subker, op.k, op.c);
	}
};

template<typename Scalar>
struct eye_umker_t : public um_ker_t<Scalar, eye_umker_t<Scalar>> {
	template<typename rArg>
	__host_device_func auto evalExp(rArg arg) {
		return arg;
	}
	__host_device_func auto eval(var_exp_t<void, Scalar>& v) {
		auto expr = evalExp(v.ref());
		expr.eval();
		expr.backward(1);
		return expr;
	}
};

template<typename Scalar, typename opExp>
struct linear_umker_t : public um_ker_t<Scalar, linear_umker_t<Scalar, opExp>> {
	using opExp_t = opExp;
	// a_ * op + b_
	opExp op;
	using Base = um_ker_t<Scalar, linear_umker_t<Scalar, opExp>>;
	Scalar a = 1, b = 0;
	linear_umker_t(opExp op_, Scalar a_, Scalar b_)
		: op(op_), a(a_), b(b_) {}
	template<typename rArg>
	__host_device_func auto evalExp(rArg arg) {
		return op.template evalExp<rArg>(arg) * a + b;
	}
	//using Exp = decltype(((linear_umker_t*)nullptr)->evalExp(*((rvar_exp_t<Scalar>*)nullptr)));
	__host_device_func auto eval(var_exp_t<void, Scalar>& v) {
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
struct pow_umker_t : public um_ker_t<Scalar, pow_umker_t<Scalar, opExp>> {
	using opExp_t = opExp;
	opExp op;
	using Base = um_ker_t<Scalar, pow_umker_t<Scalar, opExp>>;
	Scalar p;
	pow_umker_t(opExp op_, Scalar pon)
		: op(op_), p(pon) {}

	template<typename rArg>
	__host_device_func auto evalExp(rArg arg) {
		return op.template evalExp<rArg>(arg).pow(p);
	}
	//using Exp = decltype(((pow_umker_t<Scalar, opExp>*)nullptr)->evalExp(*((rvar_exp_t<Scalar>*)nullptr)));
	__host_device_func auto eval(var_exp_t<void, Scalar>& v) {
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
struct exp_umker_t : public um_ker_t<Scalar, exp_umker_t<Scalar, opExp>> {
	using opExp_t = opExp;
	opExp op;
	using Base = um_ker_t<Scalar, exp_umker_t<Scalar, opExp>>;
	exp_umker_t(opExp op_)
		: op(op_) {}

	template<typename rArg>
	__host_device_func auto evalExp(rArg arg) {
		return op.template evalExp<rArg>(arg).exp();
	}
	//using Exp = decltype(((pow_umker_t<Scalar, opExp>*)nullptr)->evalExp(*((rvar_exp_t<Scalar>*)nullptr)));
	__host_device_func auto eval(var_exp_t<void, Scalar>& v) {
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

template<typename Scalar, typename opExp = eye_umker_t<Scalar>>
struct sigmoid_umker_t : public um_ker_t<Scalar, sigmoid_umker_t<Scalar, opExp>> {
	using opExp_t = opExp;
	opExp op;
	Scalar c, k;
	using Base = um_ker_t<Scalar, sigmoid_umker_t<Scalar, opExp>>;
	sigmoid_umker_t(opExp op_ = opExp(), Scalar k_ = 30, Scalar c_ = 0.5)
		: op(op_), k(k_), c(c_) {}

	template<typename rArg>
	__host_device_func auto evalExp(rArg arg) {
		return op.template evalExp<rArg>(arg).sgm(k, c);
	}
	//using Exp = decltype(((pow_umker_t<Scalar, opExp>*)nullptr)->evalExp(*((rvar_exp_t<Scalar>*)nullptr)));
	__host_device_func auto eval(var_exp_t<void, Scalar>& v) {
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
	: public tsexp_t<unarymap_tsexp_t<Scalar, Kernel, opExp_t>, Scalar> {
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
		auto accs = op.value().view();
		auto sdif = Base::getTemp().view();
		auto accd = Base::value().view();
		auto cfg = make_kernel_param(Base::n_valid(), 256);
		unarymap_kernel<<<cfg.grid, cfg.block>>>(accs, sdif, accd, Base::n_valid(), ker);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	__host_device_func void backward_imp(Tensor<Scalar> lastdiff) {
		auto temp = Base::getTemp().view();
		auto lastdifview = lastdiff.view();
		auto src = op.value().view();
		auto srcDiff = op.diff().view();
		auto cfg = make_kernel_param(Base::n_valid(), 256);
		unarymap_backward<<<cfg.grid, cfg.block>>>(src, srcDiff, temp, lastdifview, Base::n_valid(), ker);
		cudaDeviceSynchronize();
		cuda_error_check;
		op.backward(op.diff());
	}
};

template<typename Scalar, typename WeightFunction>
struct convker_t {
	Scalar pad = 0;
	bool period = false;
	int region[3];
	WeightFunction wfunc;
	Scalar wsum;
	bool normalize;
	__host_device_func convker_t(WeightFunction func, int halfdomain[3], Scalar padding /*= 0*/, bool normalize_ /*= false*/, bool period_ /* = false*/)
		: pad(padding), wfunc(func), normalize(normalize_), period(period_) {
		region[0] = halfdomain[0];
		region[1] = halfdomain[1];
		region[2] = halfdomain[2];
		wsum = weightSum();
	}
	__host_device_func convker_t(WeightFunction func, int halfdomain, Scalar padding /* = 0*/, bool normalize_ /*= false*/, bool period_ /*= false*/)
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
	__host_device_func bool is_period(void) const {
		return period;
	}
	__host_device_func Scalar padValue() const {
		return pad;
	}
	__host_device_func void neigh(int id, int off[3]) const {
		off[0] = id % (region[0] * 2 + 1) - region[0];
		off[1] = id / (region[0] * 2 + 1) % (region[1] * 2 + 1) - region[1];
		off[2] = id / ((region[0] * 2 + 1) * (region[1] * 2 + 1)) - region[2];
	}
	__host_device_func Scalar weight(const int off[3]) const {
		Scalar w = wfunc(off);
		if (normalize) {
			w /= wsum;
		}
		//printf("wf = %f  wsum = %f\n", w, wsum);
		return w;
	}
};

template<typename WeightFunction, typename Scalar = float>
convker_t<Scalar, WeightFunction> make_convker(WeightFunction func, int halfdomain[3], Scalar padding = 0, bool normalize = true, bool period = true) {
	return convker_t<Scalar, WeightFunction>(func, halfdomain, padding, normalize, period);
}

enum RadialConvWeight {
	Spline4,
	Linear
};

template<typename Scalar, RadialConvWeight WeightType>
struct WeightFunc {};
template<typename Scalar>
struct WeightFunc<Scalar, Spline4> {
	Scalar r;
	__host_device_func WeightFunc<Scalar, Spline4>(Scalar r_)
		: r(r_) {}
	__host_device_func Scalar operator()(const int off[3]) const {
		Scalar r_off2 = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]) / (r * r);
		Scalar r_off = sqrt_(r_off2);
		Scalar val = 0;
		if (r_off2 < 1)
			val = 1 - 6 * r_off2 + 8 * r_off2 * r_off - 3 * r_off2 * r_off2;
		//printf("r2 = %f  r = %f\n", r_off2, r_off);
		return val;
	}
};
template<typename Scalar>
struct WeightFunc<Scalar, Linear> {
	Scalar r;
	__host_device_func WeightFunc<Scalar, Linear>(Scalar r_)
		: r(r_) {}
	__host_device_func Scalar operator()(const int off[3]) const {
		Scalar r_off2 = (off[0] * off[0] + off[1] * off[1] + off[2] * off[2]) / (r * r);
		Scalar r_off = sqrt_(r_off2);
		Scalar val = 0;
		if (r_off2 < 1)
			val = 1 - r_off;
		//printf("r2 = %f  r = %f\n", r_off2, r_off);
		return val;
	}
};

template<typename Scalar, RadialConvWeight wType>
struct radial_convker_t {};
template<typename Scalar>
struct radial_convker_t<Scalar, Spline4>
	: convker_t<Scalar, WeightFunc<Scalar, Spline4>> {
	__host_device_func radial_convker_t<Scalar, Spline4>(Scalar r, Scalar padding = 0, bool normalize = true, bool period = false)
		: convker_t<Scalar, WeightFunc<Scalar, Spline4>>(
			  WeightFunc<Scalar, Spline4>(r), r + 0.5, padding, normalize, period) {}
};
template<typename Scalar>
struct radial_convker_t<Scalar, Linear>
	: convker_t<Scalar, WeightFunc<Scalar, Linear>> {
	__host_device_func radial_convker_t<Scalar, Linear>(Scalar r, Scalar padding = 0, bool normalize = true, bool period = false)
		: convker_t<Scalar, WeightFunc<Scalar, Linear>>(
			  WeightFunc<Scalar, Linear>(r), r + 0.5, padding, normalize, period) {}
};

template<typename Scalar, typename Kernel, typename opExp_t>
struct conv_tsexp_t
	: public tsexp_t<conv_tsexp_t<Scalar, Kernel, opExp_t>, Scalar> {
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
		auto cfg = make_kernel_param(Base::n_valid(), 256);
		conv_kernel<<<cfg.grid, cfg.block>>>(Base::n_valid(), srcAcc, dstAcc, tmpacc, ker);
		cudaDeviceSynchronize();
		cuda_error_check;
	}

	__host_device_func void backward_imp(Tensor<Scalar> lastdiff) {
		auto dstAcc = lastdiff.view();
		auto srcAcc = op.diff().view();
		auto tmpacc = Base::getTemp().view();
		auto cfg = make_kernel_param(Base::n_valid(), 256);
		conv_backward_kernel<<<cfg.grid, cfg.block>>>(Base::n_valid(), srcAcc, dstAcc, tmpacc, ker);
		cudaDeviceSynchronize();
		cuda_error_check;
		op.backward(op.diff());
	}
};
// define full expression
struct AssociativeOpBase {};

struct Void_t {};
template<typename Wrapped>
struct Shell : Wrapped {
	Shell(const Wrapped& wr)
		: Wrapped(wr) {}
	Shell(void) = default;
};

template<typename Scalar, typename binop, typename UnaryMap, typename InvUnaryMap = inv_func_t<Scalar, UnaryMap>>
struct AssociativeOp
	: AssociativeOpBase,
	  binop,
	  Shell<UnaryMap>,
	  //  std::conditional_t<std::is_convertible_v<InvUnaryMap, UnaryMap>, Void_t, InvUnaryMap>
	  InvUnaryMap {
	using g = Shell<UnaryMap>;
	// using gInv = std::conditional_t<std::is_convertible_v<InvUnaryMap, UnaryMap>, UnaryMap, InvUnaryMap>;
	using gInv = InvUnaryMap;
	using Sop = binop;
	using Sop::combine;
	constexpr static auto Id = Sop::Identity;
	template<typename G = UnaryMap,
			 // if func has status, the inv func should have status too
			 std::enable_if_t<!std::is_empty_v<G>, int> = 0>
	AssociativeOp(const UnaryMap& f)
		: g(f), gInv(f) {}
	template<typename G = UnaryMap,
			 // if func has status, the inv func should have status too
			 std::enable_if_t<std::is_empty_v<G>, int> = 0>
	AssociativeOp(void){};

	__host_device_func Scalar merge(Scalar s, Scalar v) const {
		return gInv::valueAt(Sop::combine(g::valueAt(s), g::valueAt(v)));
	}
	__host_device_func Scalar raw_merge(Scalar s_raw, Scalar v) const {
		return Sop::combine(s_raw, g::valueAt(v));
	}
	__host_device_func Scalar preMap(Scalar v) const {
		return g::valueAt(v);
	}
	__host_device_func Scalar postMap(Scalar s) const {
		return gInv::valueAt(s);
	}
	template<typename T = binop>
	__host_device_func std::enable_if_t<std::is_same_v<T, addop_func_t<Scalar>>, Scalar> diff(Scalar s, Scalar xk) const {
		return g::diffAt(xk) / g::diffAt(s);
	}
	template<typename T = binop>
	__host_device_func std::enable_if_t<std::is_same_v<T, mulop_func_t<Scalar>>, Scalar> diff(Scalar s, Scalar xk) const {
		Scalar pr = g::valueAt(s);
		Scalar px = g::valueAt(xk);
		return g::diffAt(xk) * pr / g::diffAt(s) / px;
	}
};

template<typename Scalar>
struct SumReduceOp : public AssociativeOp<Scalar, addop_func_t<Scalar>, eye_func_t<Scalar>> {};

template<typename Scalar>
struct LogSumExpOp : public AssociativeOp<Scalar, addop_func_t<Scalar>, exp_func_t<Scalar>> {};

template<typename Scalar>
struct PnormReduceOp : public AssociativeOp<Scalar, addop_func_t<Scalar>, pow_func_t<Scalar>> {
	PnormReduceOp(Scalar p)
		: AssociativeOp<Scalar, addop_func_t<Scalar>, pow_func_t<Scalar>>(p) {}
};

template<typename Scalar, typename Operand, typename ReduceOp>
struct reduce_tsexp_t
	: public exp_t<reduce_tsexp_t<Scalar, Operand, ReduceOp>, Scalar> {
	using Base = exp_t<reduce_tsexp_t<Scalar, Operand, ReduceOp>, Scalar>;
	Operand opr;
	ReduceOp op;
	reduce_tsexp_t(const Operand& opr_, const ReduceOp& rop_)
		: opr(opr_), op(rop_) {}
	struct bracketWrapper {
		Scalar* pdata;
		__host_device_func auto& operator()(int k) {
			return pdata[k];
		}
		__host_device_func auto& operator()(int k) const {
			return pdata[k];
		}
	};
	__host_device_func Scalar eval_imp(void) {
		opr.eval();
		auto accs = opr.value().view();
		auto n_src = accs.size();
		Scalar res = 0;
		auto cfg = make_kernel_param(n_src, 512);
		auto buffer = getTempBuffer(cfg.grid * sizeof(Scalar));
		Scalar* pbuf = buffer.template data<Scalar>();
		tensor_reduce<512, false><<<cfg.grid, cfg.block>>>(n_src, accs, bracketWrapper{pbuf}, op);
		cudaDeviceSynchronize();
		cuda_error_check;
		auto buffer1 = getTempBuffer((cfg.grid / 200 + 1) * sizeof(Scalar));
		while (cfg.grid > 1) {
			auto pbuf1 = buffer1.template data<Scalar>();
			n_src = cfg.grid;
			cfg = make_kernel_param(n_src, 512);
			tensor_reduce<512, true><<<cfg.grid, cfg.block>>>(n_src, bracketWrapper{pbuf}, bracketWrapper{pbuf1}, op);
			cudaDeviceSynchronize();
			cuda_error_check;
			std::swap(pbuf, pbuf1);
		}
		cudaMemcpy(&res, pbuf, sizeof(Scalar), cudaMemcpyDeviceToHost);
		return res;
	}
	__host_device_func void backward_imp(Scalar lastdiff) {
		auto gradacc = opr.diff().view();
		int n_src = gradacc.size();
		auto cfg = make_kernel_param(n_src, 512);
		tensor_reduce_gradient<512, Scalar><<<cfg.grid, cfg.block>>>(opr.value().view(), gradacc, Base::value(), lastdiff, op);
		cudaDeviceSynchronize();
		cuda_error_check;
		opr.backward(opr.diff());
	}
};

template<typename subVar = void, typename Scalar = float>
struct var_tsexp_t
	: tsexp_t<var_tsexp_t<subVar, Scalar>, Scalar> {
	//Tensor<Scalar>& vars;
	//Tensor<Scalar>& vardiffs;
	using Base = tsexp_t<var_tsexp_t<subVar, Scalar>, Scalar>;
	//var_tsexp_t(int xlen, int ylen, int zlen) :vars(Base::values), vardiffs(Base::diffs), Base(xlen, ylen, zlen) { vardiffs.reset(); }
	__host_device_func var_tsexp_t(int xlen, int ylen, int zlen)
		: Base(xlen, ylen, zlen) {
		Base::diff().reset();
	}
	__host_device_func var_tsexp_t(Tensor<Scalar> val, Tensor<Scalar> dif)
		: Base(val, dif) {}
	__host_device_func var_tsexp_t(Tensor<Scalar> val)
		: Base(val) {}
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
template<typename T = float>
using TensorVar = var_tsexp_t<void, T>;
} // namespace homo

#pragma pop_macro("max")
#pragma pop_macro("min")
