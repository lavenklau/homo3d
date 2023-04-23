#pragma once

#include "engine.h"
#include <cstdint>
#include <cstdarg>

#include <vector>
#include <Eigen/Sparse>
#include "platform_spec.h"

// #define ENABLE_MATLAB

#pragma warning(disable:4244)
//#define MATLAB_DEFAULT_RELEASE R2017b
//#define MATLAB_DEFAULT_RELEASE R2018a

#define MATLAB_DEFAULT_RELEASE R2020a

inline void ensure(bool cond, const char *msg, ...)
{   if (!cond) { va_list args; va_start(args, msg); vfprintf(stderr, (msg+std::string("\n")).c_str(), args); va_end(args); } }

#define ensureTypeMatch(R, m, othertype) ensure(MatlabNum<R>::id == mxGetClassID(m), "Matlab type does not match " othertype)

class MatlabEngine
{
public:
    bool consoleOutput;

	MatlabEngine():eng(nullptr), consoleOutput(true)	{ }
	virtual ~MatlabEngine() { if(eng) close(); }

	// Run inside matlab: enableservice('AutomationServer', true)
	bool connect(const std::string &dir, bool closeAll=false);

	bool connected() const { return eng!=nullptr; }
    void setEnable(bool v) { if (v) connect(""); else close(); }

	void eval(const std::string &cmd);

	void close();

	void hold_on()	{	eval("hold on;");	}

	void hold_off()	{	eval("hold off;"); }

	const char *output_buffer()	{	return (*engBuffer)?engBuffer:nullptr;	}

	bool hasVar(const std::string &name)
	{
		ensure(connected(), "Not connected to Matlab!");
		mxArray *m = engGetVariable(eng, name.c_str());
        bool r = (m != nullptr);
        mxDestroyArray(m);
        return r;
	}

	mxArray* getVariable(const std::string &name)
	{
		ensure(connected(), "Not connected to Matlab!");
		mxArray *m = engGetVariable(eng, name.c_str());
        ensure(m!=nullptr, "Matlab doesn't have a variable: %s\n", name.c_str());

		return m;
	}

	int putVariable(const std::string &name, const mxArray *m)
	{	ensure(connected(), "Not connected to Matlab!");	return engPutVariable(eng, name.c_str(), m); 	}


private:
	Engine *eng; // Matlab engine
	static const int lenEngBuffer = 1000000;
	char engBuffer[lenEngBuffer]; // engine buffer for outputting strings
};

MatlabEngine& getMatEngine();
// inline void matlabEval(const char* cmd) { getMatEngine().eval(cmd); }
inline void matlabEval(const std::string &cmd) { getMatEngine().eval(cmd.c_str()); }

inline bool matEngineConnected() { return getMatEngine().connected();  }

template<typename R>
struct MatlabNum
{
	static const mxClassID id = mxUNKNOWN_CLASS;
};

template<>	struct MatlabNum<bool>	{	static const mxClassID id = mxLOGICAL_CLASS; };
template<>	struct MatlabNum<char>	{	static const mxClassID id = mxCHAR_CLASS; };
template<>	struct MatlabNum<int>	{	static const mxClassID id = mxINT32_CLASS; };
template<>	struct MatlabNum<float>	{	static const mxClassID id = mxSINGLE_CLASS; };
template<>	struct MatlabNum<double>{	static const mxClassID id = mxDOUBLE_CLASS; };

template<typename R>
inline mxArray* createMatlabArray(const mwSize *dims, int ndim)
{	return mxCreateNumericArray(ndim, dims, MatlabNum<R>::id, mxREAL);	}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<class R>
void VecVec2IdxVec(const std::vector<std::vector<R> >& in, std::vector<int>& v, std::vector<int>& idx)
{
	const size_t nvec = in.size();

	idx.resize(nvec+1);
	idx[0] = 0;

	size_t n = 0;
	for(size_t i=0; i<nvec; i++) n += in[i].size();
	v.clear(); 
	v.reserve(n);

	for(size_t i=0; i<nvec; i++){
		v.insert(v.end(), in[i].begin(), in[i].end());
		idx[i+1] = idx[i]+int(in[i].size());
	}
}

template<class R>
std::vector<std::vector<R> > IdxVec2VecVec(const std::vector<int>& vcat, const std::vector<int>& vidx)
{
	ensure(!vidx.empty(), "empty indices");

	const size_t nvec = vidx.size()-1;
	std::vector<std::vector<R> > v(nvec);

	for(size_t i=0; i<nvec; i++) v[i] = std::vector<R>(vcat.begin()+vidx[i], vcat.begin()+vidx[i+1]);

	return v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template <class R>
void vector2matlab(const std::vector<R> &v, mxArray *m)
{
	ensureTypeMatch(R, m, "std::vector"); 
	R *pm = (R *)mxGetData(m);
	for ( unsigned i = 0 ; i < v.size() ; i++ )
		pm[i] = (R)v[i];
}


template<typename R=double>
void vector2matlab(const std::string &name , const std::vector<R> &v)
{
	mwSize dim[] = { v.size() };
	mxArray *m = createMatlabArray<R>(dim, 1);

	vector2matlab(v, m);
	getMatEngine().putVariable(name, m);
	mxDestroyArray(m);
}

template <class R>
std::vector<R> matlab2vector(const mxArray *m)
{
	ensureTypeMatch(R, m, "std::vector"); 
    if (mxIsSparse(m)){ ensure(false, "matrix is sparse!"); return std::vector<R>(); }

	const R *pm = (R*)mxGetData(m);
	return std::vector<R>( pm, pm + mxGetNumberOfElements(m) );
}

inline std::string matlab2string(const std::string &name)
{
	mxArray *m = getMatEngine().getVariable(name);
    //std::wstring str(mxGetChars(m));
    if (!m) return std::string();

    size_t len = mxGetNumberOfElements(m) + 1;
    std::vector<char> str(len);
    mxGetString(m, str.data(), len);
    mxDestroyArray(m);
    return std::string(str.cbegin(), str.cend()-1);
}

inline std::vector<std::string> matlab2strings(const std::string &name)
{
    mxArray *m = getMatEngine().getVariable(name);
    std::vector<std::string> strs;
    if (!m) return strs;

    for (int i = 0; i < mxGetNumberOfElements(m); i++) {
        mxArray *mstr = mxGetCell(m, i);

        size_t len = mxGetNumberOfElements(mstr) + 1; // for \0 end
        std::vector<char> str(len);
        mxGetString(mstr, str.data(), len);
        strs.push_back(std::string(str.cbegin(), str.cend()-1));
    }

    mxDestroyArray(m);
    return strs;
}

inline void string2matlab(const std::string &name, const std::string &val)
{
    mxArray *m = mxCreateString(val.c_str());
    getMatEngine().putVariable(name, m);
    mxDestroyArray(m);
}


template <class R=double>
std::vector<R> matlab2vector(const std::string &name, bool temp=false)
{
    const std::string tempname("mytempval4c");
    if (temp)
        getMatEngine().eval(tempname + "=" + name + ";");

	mxArray *m = getMatEngine().getVariable(temp?tempname:name);
    if (!m)  return std::vector<R>();

	std::vector<R> v = matlab2vector<R>(m);

	mxDestroyArray(m);

    if (temp)
        getMatEngine().eval( "clear " + tempname + ";");

	return v;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
template<class R=double>
void vecvec2matlabcell(const std::string &name, const std::vector<std::vector<R> > &v)
{
	std::vector<int> vcat, vidx;
	VecVec2IdxVec(v, vcat, vidx);
	vector2matlab("vcat_tmp", vcat);
	vector2matlab("vidx_tmp", vidx);

	std::stringstream ss;
	ss<<name<<" = indexedArray2cell(vcat_tmp+1, vidx_tmp+1); clear vcat_tmp vidx_tmp;";

	matlabEval( ss.str() );
}

inline bool incMatCell(const std::string &name)
{
	std::stringstream ss;
	ss<<name<<" = cellfun( @(x) x+1, "<<name<<", 'UniformOutput', false);";
	matlabEval( ss.str() );
}

inline bool decMatCell(const std::string &name)
{
	std::stringstream ss;
	ss<<name<<" = cellfun( @(x) x-1, "<<name<<", 'UniformOutput', false);";
	matlabEval( ss.str() );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
template<class R>
void matlabcell2idxvec(const std::string &name, std::vector<R> &vcat, std::vector<R> &vidx)
{
	std::stringstream ss;
	ss<<"[vcat_tmp vidx_tmp]=cell2indexedArray("<<name<<");";

	matlabEval( ss.str() );
	vcat = matlab2vector<int>("vcat_tmp");
	vidx = matlab2vector<int>("vidx_tmp");
	matlabEval( "clear vcat_tmp vidx_tmp;" );
}

//template<class R>
//std::vector<std::vector<R> > matlabcell2vecvec(const std::string &name)
//{
//	std::vector<int> vcat, vidx;
//	matlabcell2vecvec(name, vcat, vidx);
//	return IdxVec2VecVec(vcat, vidx);
//}
//////////////////////////////////////////////////////////////////////////////////////////////////////
//template <class M>
//void matlab2eigen(const mxArray *m , Eigen::MatrixBase<M> &v)
//{
//	typedef M::Scalar R;
//	const mwSize *dim = mxGetDimensions(m);
//	const R *pm = (R*)mxGetData(m);
//
//	ensure(dim[0]==v.rows() && dim[1]==v.cols());
//	//v.resize(dim[0], dim[1]);
//	 
//	for ( unsigned i = 0 ; i < dim[0] ; i++ )
//		for ( unsigned j = 0 ; j < dim[1] ; j++ ) {
//			const mwSize ind2[] = {i, j};
//			v(i,j) = pm[mxCalcSingleSubscript(m, 2, ind2)];
//		}
//}
//
//template <class M>
//void matlab2eigen(const std::string &name, Eigen::MatrixBase<M> &v)
//{
//	mxArray *m = getMatEngine().getVariable(name);
//	if ( !m ) 	return;
//
//	matlab2eigen(m, v);
//
//	mxDestroyArray(m);
//}

template <class EigenMatrix>
void matlab2eigen(const mxArray *m , EigenMatrix &v)
{
	typedef typename EigenMatrix::Scalar R;
	ensureTypeMatch(R, m, "Eigen::Matrix"); 
    if (mxIsSparse(m)){ ensure(false, "matrix is sparse!"); return; }

	const mwSize *dim = mxGetDimensions(m);

	v = Eigen::Map<const Eigen::Matrix<R,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> >((R*)mxGetData(m), dim[0], dim[1]);
}

template <class Matrix>
void matlab2eigen(const std::string &name, Matrix &v, bool temp=false)
{
    const std::string tempname("mytempval4c");
    if (temp)
        getMatEngine().eval(tempname + "=" + name + ";");

	mxArray *m = getMatEngine().getVariable(temp?tempname:name);
    if (!m)  return;

	matlab2eigen(m, v);

	mxDestroyArray(m);

    if (temp)
        getMatEngine().eval( "clear " + tempname + ";");
}

template<class eigenSpMatrix>
void matlabSpmat2eigen(const mxArray* Spm, eigenSpMatrix & v)
{
	typedef typename eigenSpMatrix::Scalar R;
	typedef Eigen::Map<Eigen::SparseMatrix<R, Eigen::ColMajor, std::make_signed<mwIndex>::type>> EigenSparseMat;
	ensureTypeMatch(R, Spm, "Eigen::SparseMatrix");
	if (!mxIsSparse(Spm)) { ensure(false, "matrix is not sparse!"); return; }

	const mwSize *dim = mxGetDimensions(Spm);
	typename EigenSparseMat::StorageIndex* Ir = reinterpret_cast<typename EigenSparseMat::StorageIndex*>(mxGetIr(Spm));
	typename EigenSparseMat::StorageIndex* Jc = reinterpret_cast<typename EigenSparseMat::StorageIndex*>(mxGetJc(Spm));
	R* Vp = reinterpret_cast<R*>(mxGetPr(Spm));
	auto nnzs = mxGetNzmax(Spm);
	int nrows = mxGetM(Spm);
	int ncols = mxGetN(Spm);

	v = Eigen::Map<Eigen::SparseMatrix<R, Eigen::ColMajor, std::make_signed<mwIndex>::type>>(nrows, ncols, nnzs, Jc, Ir, Vp);
}

template<class eigenSpMatrix>
void matlabSpmat2eigen(const std::string &name, eigenSpMatrix & v) {
	auto& eng = getMatEngine();
	if (!eng.connected()) {
		eng.connect("");
	}
	auto* m = eng.getVariable(name);
	if (m == nullptr) {
		printf("matrix %s is not in MATLAB!", name.c_str());
		return;
	}
	matlabSpmat2eigen(m, v);
	mxDestroyArray(m);
}


template<class Mat>
inline void eigen2matlabComplex(const std::string &name, const Mat &vr, const Mat &vi)
{
    mwSize dim[] = { vr.rows(), vr.cols() };
    mxArray *m = mxCreateNumericArray(2, dim, MatlabNum<double>::id, mxCOMPLEX);

    using MapMat = Eigen::Map < Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> > ;
    MapMat(mxGetPr(m), dim[0], dim[1]) = vr;
    MapMat(mxGetPi(m), dim[0], dim[1]) = vi;

    getMatEngine().putVariable(name, m);
    mxDestroyArray(m);
}

template <class M>
void eigen2matlab(const Eigen::MatrixBase<M> &v, mxArray *m)
{
	typedef typename M::Scalar R;
	ensureTypeMatch(R, m, "Eigen::Matrix"); 

    using namespace Eigen;
    Map<Matrix<R, Dynamic, Dynamic, ColMajor> >((R*)mxGetData(m), v.rows(), v.cols()) = v;
}

inline void scalar2matlab(const std::string &name, double v) {
    mxArray *m = mxCreateDoubleScalar(v);
    getMatEngine().putVariable(name, m);
	mxDestroyArray(m);
}

inline double matlab2scalar(const std::string &name, double fallback=0, bool temp=false) {
    const std::string tempname("mytempval4c");

    auto &eng = getMatEngine();
    if (temp)  eng.eval(tempname + "=" + name + ";");

    if (!eng.hasVar(temp ? tempname : name)) return fallback;

	mxArray *m = eng.getVariable(temp?tempname:name);
    ensure(mxIsScalar(m), "Matlab: %s is not a scalar!", name.c_str());

    double r = (m && mxIsScalar(m))?mxGetScalar(m):fallback;
    mxDestroyArray(m);

    if (temp)
        eng.eval( "clear " + tempname + ";");

    return r; 
}

template<class M>
void eigen2matlab(const std::string &name, const Eigen::MatrixBase<M> &v)
{
	typedef typename M::Scalar R;
	mwSize dim[] = { mwSize(v.rows()), mwSize(v.cols()) };
	mxArray *m = createMatlabArray<R>(dim, 2);
	if (m == nullptr) printf("-- \033[31meigen2mat failed (nullptr)\033[0m\n");
	eigen2matlab(v, m);
	if (getMatEngine().putVariable(name, m)) {
		printf("-- \033[31meigen2mat failed (transfer failed)\033[0m\n");
	}
	mxDestroyArray(m);
}


template <class R>
void eigen2matlab(const std::string &name, const Eigen::SparseMatrix<R,Eigen::ColMajor> &A)
{
    int rows = A.rows(), cols = A.cols(), nnz = A.nonZeros();
    mxArray *m = mxCreateSparse(rows, cols, nnz, mxREAL);
	if (m == nullptr) printf("-- \033[31meigen2mat failed (nullptr)\033[0m\n");

	//mxSetPr(m, (double*)mxRealloc(mxGetPr(m), (nnz + 1) * sizeof(R)));
	//mxSetIr(m, (mwIndex*)mxRealloc(mxGetIr(m), (nnz + 1) * sizeof(mwIndex)));

    std::copy_n(A.valuePtr(), nnz, mxGetPr(m));
    std::copy_n(A.outerIndexPtr(), rows+1, mxGetJc(m));
    std::copy_n(A.innerIndexPtr(), nnz, mxGetIr(m));

	if (getMatEngine().putVariable(name, m)) {
		printf("-- \033[31meigen2mat failed (transfer failed)\033[0m\n");
	}
    mxDestroyArray(m);
}

template<class M>
void eigen2ConnectedMatlab(const std::string &name, const Eigen::MatrixBase<M> &v) {
#ifdef ENABLE_MATLAB
	auto& eng = getMatEngine();
	if (!eng.connected())
	{
		eng.connect("");
	}
	eigen2matlab(name, v);
#endif
}

template<class R>
void array2ConnectedMatlab(const std::string& name, const R* ptr, size_t len) {
#ifdef ENABLE_MATLAB
	auto& eng = getMatEngine();
	if (!eng.connected()) {
		eng.connect("");
	}
	Eigen::Matrix<double, -1, 1> x(len, 1);
	for (int i = 0; i < len; i++) {
		x[i] = ptr[i];
	}
	eigen2matlab(name, x);
#endif
}

template<class R>
void arrays2ConnectedMatlab(const std::string& name, const R** ptr, int len, int n_array) {
#ifdef ENABLE_MATLAB
	auto& eng = getMatEngine();
	if (!eng.connected()) {
		eng.connect("");
	}
	Eigen::Matrix<double, -1, -1> x(len, n_array);
	for (int n = 0; n < n_array; n++) {
		for (int i = 0; i < len; i++) {
			x(i, n) = ptr[n][i];
		}
	}
	eigen2matlab(name, x);
#endif
}

template <class R>
void eigen2ConnectedMatlab(const std::string &name, const Eigen::SparseMatrix<R, Eigen::ColMajor> &A) {
#ifdef ENABLE_MATLAB
	auto& eng = getMatEngine();
	if (!eng.connected())
	{
		eng.connect("");
	}
	if (A.cols() != A.rows()) {
		int maxsize = 0;
		if (A.cols() != A.rows()) {
			maxsize = (std::max)(A.cols(), A.rows());
		}
		Eigen::SparseMatrix<R, Eigen::ColMajor> B = A;
		B.conservativeResize(maxsize, maxsize);
		eigen2matlab(name, B);
		Eigen::Matrix<double, 2, 1> bsize;
		bsize[0] = A.rows(); bsize[1] = A.cols();
		eigen2ConnectedMatlab("asize", bsize);
		char resizestr[1000];
		sprintf_s(resizestr, "%s=%s(1:asize(1),1:asize(2));", name.c_str(), name.c_str());
		eng.eval(resizestr);
	}
	else {
		eigen2matlab(name, A);
	}
#endif
}

template<class R>
void eigen2ConnectedMatlab(const std::string &name, const Eigen::SparseMatrix<R, Eigen::RowMajor> &A) {
#ifdef ENABLE_MATLAB
	auto& eng = getMatEngine();
	if (!eng.connected()) {
		eng.connect("");
	}
	Eigen::SparseMatrix<R, Eigen::ColMajor> Ac = A;
	eigen2matlab(name, Ac);
#endif
}

template<typename T, typename std::enable_if<std::is_scalar<T>::value, int >::type = 0>
void scaler2ConnectedMatlab(const std::string &name, T val) {
#ifdef ENABLE_MATLAB
	Eigen::Matrix<double, 1, 1> scalarmat;
	scalarmat[0] = val;
	eigen2ConnectedMatlab(name, scalarmat);
#endif
}
