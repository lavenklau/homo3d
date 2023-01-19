/*
 * External header for the libmx library.
 *
 * Copyright 1984-2018 The MathWorks, Inc.
 * All Rights Reserved.
 */

#ifdef MDA_ARRAY_HPP_
#error Using MATLAB Data API with C Matrix API is not supported.
#endif

#if defined(_MSC_VER)
#pragma once
#endif
#if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ > 3))
#pragma once
#endif

#ifndef matrix_h
#define matrix_h

#include <stdlib.h>
#include <stddef.h>
#include "tmwtypes.h"

#ifdef __cplusplus
#define LIBMMWMATRIX_PUBLISHED_API_EXTERN_C extern "C"
#else
#define LIBMMWMATRIX_PUBLISHED_API_EXTERN_C extern
#endif



#define MW_FIRST_API_VERSION 700
#define R2017b 700
#define R2018a 800
#define R2018b 800
#define R2019a 800
#define R2019b 800
#define R201aa 800
#define R201ab 800
#define R201ba 800
#define R201bb 800
#define R201ca 800
#define R201cb 800
#define R201da 800
#define R201db 800
#define R201ea 800
#define R201eb 800
#define R201fa 800
#define R201fb 800
#define R2020a 800
#define R2020b 800
#define R2021a 800
#define R2021b 800
#define R2022a 800
#define MW_LATEST_API_VERSION 800


#define MW_REL2VER(A) A 

#if defined(MX_COMPAT_32) || defined(MEX_DOUBLE_HANDLE)

#if defined(MATLAB_MEXCMD_RELEASE) || defined(MATLAB_MEXSRC_RELEASE)

/* Errors! Legacy knobs cannot be used with release-based hard knobs */

#if defined(MX_COMPAT_32) && defined(MATLAB_MEXCMD_RELEASE)
#error "MEX command option -R20XXx is incompatible with MX_COMPAT_32"
#endif

#if defined(MEX_DOUBLE_HANDLE) && defined(MATLAB_MEXCMD_RELEASE)
#error "MEX command option -R20XXx is incompatible with MEX_DOUBLE_HANDLE"
#endif

#if defined(MX_COMPAT_32) && defined(MATLAB_MEXSRC_RELEASE)
#error "Source code macro MATLAB_MEXSRC_RELEASE is incompatible with MX_COMPAT_32"
#endif

#if defined(MEX_DOUBLE_HANDLE) && defined(MATLAB_MEXSRC_RELEASE)
#error "Source code macro MATLAB_MEXSRC_RELEASE is incompatible with MEX_DOUBLE_HANDLE"
#endif

#else

/* Legacy knobs are defined  */

#define MATLAB_TARGET_API_VERSION MW_FIRST_API_VERSION

#endif

#else /* defined(MX_COMPAT_32) || defined(MEX_DOUBLE_HANDLE) */

/* No Legacy knobs. Check release-based tag */

#if defined(MATLAB_MEXCMD_RELEASE)
#define MW_MEXCMD_VERSION MW_REL2VER(MATLAB_MEXCMD_RELEASE)
#if MW_MEXCMD_VERSION < MW_FIRST_API_VERSION
#error invalid MATLAB_MEXCMD_RELEASE definition
#endif
#endif

#if defined(MATLAB_MEXSRC_RELEASE)
#define MW_MEXSRC_VERSION MW_REL2VER(MATLAB_MEXSRC_RELEASE)
#if MW_MEXSRC_VERSION < MW_FIRST_API_VERSION
#error invalid MATLAB_MEXSRC_RELEASE definition
#endif
#endif
      
#if defined(MATLAB_DEFAULT_RELEASE)
#define MW_DEFAULT_VERSION MW_REL2VER(MATLAB_DEFAULT_RELEASE)
#if MW_DEFAULT_VERSION < MW_FIRST_API_VERSION
#error invalid MATLAB_DEFAULT_RELEASE definition
#endif
#endif

#if defined(MATLAB_MEXCMD_RELEASE) && defined(MATLAB_MEXSRC_RELEASE)
#if MW_MEXCMD_VERSION != MW_MEXSRC_VERSION
#error "MEX command option -R20XXx is incompatible with MATLAB_MEXSRC_RELEASE"
#endif
#endif

#if defined(MATLAB_MEXCMD_RELEASE) || defined(MATLAB_MEXSRC_RELEASE)

/* Check whether MEXCMD and MEXSRC release tags are compatible */

#if defined(MATLAB_MEXCMD_RELEASE)
#define MATLAB_TARGET_API_VERSION MW_MEXCMD_VERSION
#else
#define MATLAB_TARGET_API_VERSION MW_MEXSRC_VERSION
#endif

#else /* defined(MATLAB_MEXCMD_RELEASE) || defined(MATLAB_MEXSRC_RELEASE) */

#if defined(MATLAB_DEFAULT_RELEASE)
#define MATLAB_TARGET_API_VERSION MW_DEFAULT_VERSION
#else

/* None of the input macros are defined. Use LATEST */
#define MATLAB_TARGET_API_VERSION MW_LATEST_API_VERSION

#endif /* defined(MATLAB_DEFAULT_RELEASE) */

#endif /* defined(MATLAB_MEXCMD_RELEASE) || defined(MATLAB_MEXSRC_RELEASE) */

#endif /* defined(MX_COMPAT_32) || defined(MEX_DOUBLE_HANDLE) */

#if defined(TARGET_API_VERSION)
#if MATLAB_TARGET_API_VERSION != TARGET_API_VERSION
#error MATLAB_TARGET_API_VERSION != TARGET_API_VERSION
#endif
#else
#define TARGET_API_VERSION MATLAB_TARGET_API_VERSION
#endif



/**
 * Forward declaration for mxArray
 */
typedef struct mxArray_tag mxArray;

/**
 * MEX-file entry point type
 */
typedef void (*mxFunctionPtr)(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/**
 * Maximum mxArray name length
 */
#define mxMAXNAM TMW_NAME_LENGTH_MAX

/**
 * Logical type
 */
typedef bool mxLogical;

/**
 * Required for Unicode support in MATLAB
 */
typedef CHAR16_T mxChar;

/**
 * mxArray classes.
 */
#if !defined(__cplusplus) || __cplusplus < 201103L || defined(SWIG)
typedef enum
#else
enum mxClassID : int
#endif
{
    mxUNKNOWN_CLASS = 0,
    mxCELL_CLASS,
    mxSTRUCT_CLASS,
    mxLOGICAL_CLASS,
    mxCHAR_CLASS,
    mxVOID_CLASS,
    mxDOUBLE_CLASS,
    mxSINGLE_CLASS,
    mxINT8_CLASS,
    mxUINT8_CLASS,
    mxINT16_CLASS,
    mxUINT16_CLASS,
    mxINT32_CLASS,
    mxUINT32_CLASS,
    mxINT64_CLASS,
    mxUINT64_CLASS,
    mxFUNCTION_CLASS,
    mxOPAQUE_CLASS,
    mxOBJECT_CLASS,
#if defined(_LP64) || defined(_WIN64)
    mxINDEX_CLASS = mxUINT64_CLASS
#else
    mxINDEX_CLASS = mxUINT32_CLASS
#endif
}
#if !defined(__cplusplus) || __cplusplus < 201103L || defined(SWIG)
    mxClassID
#endif
    ;

/**
 * Indicates whether floating-point mxArrays are real or complex.
 */
typedef enum { mxREAL, mxCOMPLEX } mxComplexity;

/*
 * MATRIX numeric real data types
 */
typedef double mxDouble;
typedef float mxSingle;
typedef int8_T mxInt8;
typedef uint8_T mxUint8;
typedef int16_T mxInt16;
typedef uint16_T mxUint16;
typedef int32_T mxInt32;
typedef uint32_T mxUint32;
typedef int64_T mxInt64;
typedef uint64_T mxUint64;

#if TARGET_API_VERSION >= 800
/*
 * MATRIX numeric complex data types
 */
typedef struct { mxDouble real, imag; } mxComplexDouble;
typedef struct { mxSingle real, imag; } mxComplexSingle;
typedef struct { mxInt8 real, imag; } mxComplexInt8;
typedef struct { mxUint8 real, imag; } mxComplexUint8;
typedef struct { mxInt16 real, imag; } mxComplexInt16;
typedef struct { mxUint16 real, imag; } mxComplexUint16;
typedef struct { mxInt32 real, imag; } mxComplexInt32;
typedef struct { mxUint32 real, imag; } mxComplexUint32;
typedef struct { mxInt64 real, imag; } mxComplexInt64;
typedef struct { mxUint64 real, imag; } mxComplexUint64;

#endif /* TARGET_API_VERSION >= 800 */

#if defined(TARGET_API_VERSION)
#if !(TARGET_API_VERSION == 700 || TARGET_API_VERSION == 800)
#error invalid TARGET_VERSION_API definition
#elif defined(MEX_DOUBLE_HANDLE) && TARGET_API_VERSION != 700
#error It is illegal to use MEX_DOUBLE_HANDLE with linear versioning
#elif defined(MX_COMPAT_32) && TARGET_API_VERSION != 700
#error It is illegal to use MX_COMPAT_32 with linear versioning
#endif
#endif


#if !defined(TARGET_API_VERSION) || TARGET_API_VERSION == 700

/*
 * PUBLISHED APIs with changes in MATLAB 7.3
 */

#if !defined(MX_COMPAT_32)

#define mxSetProperty mxSetProperty_730
#define mxGetProperty mxGetProperty_730
#define mxSetField mxSetField_730
#define mxSetFieldByNumber mxSetFieldByNumber_730
#define mxGetFieldByNumber mxGetFieldByNumber_730
#define mxGetField mxGetField_730
#define mxCreateStructMatrix mxCreateStructMatrix_730
#define mxCreateCellMatrix mxCreateCellMatrix_730
#define mxCreateCharMatrixFromStrings mxCreateCharMatrixFromStrings_730
#define mxGetString mxGetString_730
#define mxGetNumberOfDimensions mxGetNumberOfDimensions_730
#define mxGetDimensions mxGetDimensions_730
#define mxSetDimensions mxSetDimensions_730
#define mxSetIr mxSetIr_730
#define mxGetIr mxGetIr_730
#define mxSetJc mxSetJc_730
#define mxGetJc mxGetJc_730
#define mxCreateStructArray mxCreateStructArray_730
#define mxCreateCharArray mxCreateCharArray_730
#define mxCreateNumericArray mxCreateNumericArray_730
#define mxCreateCellArray mxCreateCellArray_730
#define mxCreateLogicalArray mxCreateLogicalArray_730
#define mxGetCell mxGetCell_730
#define mxSetCell mxSetCell_730
#define mxSetNzmax mxSetNzmax_730
#define mxSetN mxSetN_730
#define mxSetM mxSetM_730
#define mxGetNzmax mxGetNzmax_730
#define mxCreateDoubleMatrix mxCreateDoubleMatrix_730
#define mxCreateNumericMatrix mxCreateNumericMatrix_730
#define mxCreateLogicalMatrix mxCreateLogicalMatrix_730
#define mxCreateSparse mxCreateSparse_730
#define mxCreateSparseLogicalMatrix mxCreateSparseLogicalMatrix_730
#define mxGetNChars mxGetNChars_730
#define mxCreateStringFromNChars mxCreateStringFromNChars_730
#define mxCalcSingleSubscript mxCalcSingleSubscript_730
#define mxGetDimensions_fcn mxGetDimensions_730

#else /* MX_COMPAT_32 */

/*
 * 32-bit compatibility APIs
 */

#define mxGetNumberOfDimensions mxGetNumberOfDimensions_700
#define mxGetDimensions mxGetDimensions_700
#define mxGetDimensions_fcn mxGetDimensions_700
#define mxGetIr mxGetIr_700
#define mxGetJc mxGetJc_700
#define mxGetCell mxGetCell_700
#define mxGetNzmax mxGetNzmax_700
#define mxSetNzmax mxSetNzmax_700
#define mxGetFieldByNumber mxGetFieldByNumber_700
#define mxSetProperty mxSetProperty_700
#define mxGetProperty mxGetProperty_700
#define mxSetField mxSetField_700
#define mxSetFieldByNumber mxSetFieldByNumber_700
#define mxGetField mxGetField_700
#define mxCreateStructMatrix mxCreateStructMatrix_700
#define mxCreateCellMatrix mxCreateCellMatrix_700
#define mxCreateCharMatrixFromStrings mxCreateCharMatrixFromStrings_700
#define mxGetString mxGetString_700
#define mxSetDimensions mxSetDimensions_700
#define mxSetIr mxSetIr_700
#define mxSetJc mxSetJc_700
#define mxCreateStructArray mxCreateStructArray_700
#define mxCreateCharArray mxCreateCharArray_700
#define mxCreateNumericArray mxCreateNumericArray_700
#define mxCreateCellArray mxCreateCellArray_700
#define mxCreateLogicalArray mxCreateLogicalArray_700
#define mxSetCell mxSetCell_700
#define mxSetN mxSetN_700
#define mxSetM mxSetM_700
#define mxCreateDoubleMatrix mxCreateDoubleMatrix_700
#define mxCreateNumericMatrix mxCreateNumericMatrix_700
#define mxCreateLogicalMatrix mxCreateLogicalMatrix_700
#define mxCreateSparse mxCreateSparse_700
#define mxCreateSparseLogicalMatrix mxCreateSparseLogicalMatrix_700
#define mxGetNChars mxGetNChars_700
#define mxCreateStringFromNChars mxCreateStringFromNChars_700
#define mxCalcSingleSubscript mxCalcSingleSubscript_700

#endif /* #ifdef MX_COMPAT_32 */


#elif TARGET_API_VERSION == 800

#define mxMalloc mxMalloc_800
#define mxCalloc mxCalloc_800
#define mxRealloc mxRealloc_800
#define mxGetM mxGetM_800
#define mxGetN mxGetN_800
#define mxGetNumberOfElements mxGetNumberOfElements_800
#define mxFree mxFree_800
#define mxGetEps mxGetEps_800
#define mxGetInf mxGetInf_800
#define mxGetFieldNameByNumber mxGetFieldNameByNumber_800
#define mxGetClassID mxGetClassID_800
#define mxIsNumeric mxIsNumeric_800
#define mxIsCell mxIsCell_800
#define mxIsLogical mxIsLogical_800
#define mxIsChar mxIsChar_800
#define mxIsStruct mxIsStruct_800
#define mxIsSparse mxIsSparse_800
#define mxIsDouble mxIsDouble_800
#define mxIsSingle mxIsSingle_800
#define mxIsInt8 mxIsInt8_800
#define mxIsUint8 mxIsUint8_800
#define mxIsInt16 mxIsInt16_800
#define mxIsUint16 mxIsUint16_800
#define mxIsInt32 mxIsInt32_800
#define mxIsUint32 mxIsUint32_800
#define mxIsInt64 mxIsInt64_800
#define mxIsUint64 mxIsUint64_800
#define mxIsFromGlobalWS mxIsFromGlobalWS_800
#define mxIsEmpty mxIsEmpty_800
#define mxGetFieldNumber mxGetFieldNumber_800
#define mxGetNumberOfFields mxGetNumberOfFields_800
#define mxGetClassName mxGetClassName_800
#define mxIsClass mxIsClass_800
#define mxDestroyArray mxDestroyArray_800
#define mxCreateDoubleScalar mxCreateDoubleScalar_800
#define mxCreateString mxCreateString_800
#define mxAddField mxAddField_800
#define mxRemoveField mxRemoveField_800
#define mxGetNaN mxGetNaN_800
#define mxIsFinite mxIsFinite_800
#define mxIsInf mxIsInf_800
#define mxIsNaN mxIsNaN_800
#define mxIsScalar mxIsScalar_800
#define mxIsOpaque mxIsOpaque_800
#define mxIsFunctionHandle mxIsFunctionHandle_800
#define mxIsObject mxIsObject_800
#define mxGetChars mxGetChars_800
#define mxGetUserBits mxGetUserBits_800
#define mxSetUserBits mxSetUserBits_800
#define mxSetFromGlobalWS mxSetFromGlobalWS_800
#define mxCreateUninitNumericMatrix mxCreateUninitNumericMatrix_800
#define mxCreateUninitNumericArray mxCreateUninitNumericArray_800
#define mxGetLogicals mxGetLogicals_800
#define mxCreateLogicalScalar mxCreateLogicalScalar_800
#define mxIsLogicalScalar mxIsLogicalScalar_800
#define mxIsLogicalScalarTrue mxIsLogicalScalarTrue_800
#define mxArrayToString mxArrayToString_800
#define mxArrayToUTF8String mxArrayToUTF8String_800
#define mxSetClassName mxSetClassName_800
#define mxGetNumberOfDimensions mxGetNumberOfDimensions_800
#define mxGetDimensions mxGetDimensions_800
#define mxGetIr mxGetIr_800
#define mxGetJc mxGetJc_800
#define mxGetNzmax mxGetNzmax_800
#define mxGetFieldByNumber mxGetFieldByNumber_800
#define mxGetCell mxGetCell_800
#define mxSetIr mxSetIr_800
#define mxSetJc mxSetJc_800
#define mxCalcSingleSubscript mxCalcSingleSubscript_800
#define mxSetCell mxSetCell_800
#define mxSetFieldByNumber mxSetFieldByNumber_800
#define mxGetField mxGetField_800
#define mxSetField mxSetField_800
#define mxGetProperty mxGetProperty_800
#define mxSetProperty mxSetProperty_800
#define mxCreateNumericMatrix mxCreateNumericMatrix_800
#define mxCreateNumericArray mxCreateNumericArray_800
#define mxCreateCharArray mxCreateCharArray_800
#define mxCreateDoubleMatrix mxCreateDoubleMatrix_800
#define mxCreateSparse mxCreateSparse_800
#define mxGetString mxGetString_800
#define mxCreateCharMatrixFromStrings mxCreateCharMatrixFromStrings_800
#define mxCreateCellMatrix mxCreateCellMatrix_800
#define mxCreateCellArray mxCreateCellArray_800
#define mxCreateStructMatrix mxCreateStructMatrix_800
#define mxCreateStructArray mxCreateStructArray_800
#define mxCreateLogicalArray mxCreateLogicalArray_800
#define mxCreateLogicalMatrix mxCreateLogicalMatrix_800
#define mxCreateSparseLogicalMatrix mxCreateSparseLogicalMatrix_800
#define mxCreateStringFromNChars mxCreateStringFromNChars_800
#define mxGetNChars mxGetNChars_800
#define mxSetM mxSetM_800
#define mxSetN mxSetN_800
#define mxSetDimensions mxSetDimensions_800
#define mxSetNzmax mxSetNzmax_800
#define mxGetData mxGetData_800
#define mxSetData mxSetData_800
#define mxIsComplex mxIsComplex_800
#define mxGetScalar mxGetScalar_800
#define mxGetPr mxGetPr_800
#define mxSetPr mxSetPr_800
#define mxGetElementSize mxGetElementSize_800
#define mxDuplicateArray mxDuplicateArray_800
#define mxGetDoubles mxGetDoubles_800
#define mxSetDoubles mxSetDoubles_800
#define mxGetComplexDoubles mxGetComplexDoubles_800
#define mxSetComplexDoubles mxSetComplexDoubles_800
#define mxGetSingles mxGetSingles_800
#define mxSetSingles mxSetSingles_800
#define mxGetComplexSingles mxGetComplexSingles_800
#define mxSetComplexSingles mxSetComplexSingles_800
#define mxGetInt8s mxGetInt8s_800
#define mxSetInt8s mxSetInt8s_800
#define mxGetComplexInt8s mxGetComplexInt8s_800
#define mxSetComplexInt8s mxSetComplexInt8s_800
#define mxGetUint8s mxGetUint8s_800
#define mxSetUint8s mxSetUint8s_800
#define mxGetComplexUint8s mxGetComplexUint8s_800
#define mxSetComplexUint8s mxSetComplexUint8s_800
#define mxGetInt16s mxGetInt16s_800
#define mxSetInt16s mxSetInt16s_800
#define mxGetComplexInt16s mxGetComplexInt16s_800
#define mxSetComplexInt16s mxSetComplexInt16s_800
#define mxGetUint16s mxGetUint16s_800
#define mxSetUint16s mxSetUint16s_800
#define mxGetComplexUint16s mxGetComplexUint16s_800
#define mxSetComplexUint16s mxSetComplexUint16s_800
#define mxGetInt32s mxGetInt32s_800
#define mxSetInt32s mxSetInt32s_800
#define mxGetComplexInt32s mxGetComplexInt32s_800
#define mxSetComplexInt32s mxSetComplexInt32s_800
#define mxGetUint32s mxGetUint32s_800
#define mxSetUint32s mxSetUint32s_800
#define mxGetComplexUint32s mxGetComplexUint32s_800
#define mxSetComplexUint32s mxSetComplexUint32s_800
#define mxGetInt64s mxGetInt64s_800
#define mxSetInt64s mxSetInt64s_800
#define mxGetComplexInt64s mxGetComplexInt64s_800
#define mxSetComplexInt64s mxSetComplexInt64s_800
#define mxGetUint64s mxGetUint64s_800
#define mxSetUint64s mxSetUint64s_800
#define mxGetComplexUint64s mxGetComplexUint64s_800
#define mxSetComplexUint64s mxSetComplexUint64s_800
#define mxMakeArrayReal mxMakeArrayReal_800
#define mxMakeArrayComplex mxMakeArrayComplex_800
#define mxGetPi mxGetPiIsDeprecated
#define mxGetImagData mxGetImagDataIsDeprecated
#define mxSetImagData mxSetImagDataIsDeprecated
#define mxSetPi mxSetPiIsDeprecated
#define mxCreateSharedDataCopy mxCreateSharedDataCopyIsDeprecated
#define mxCreateUninitDoubleMatrix mxCreateUninitDoubleMatrixIsDeprecated
#define mxFastZeros mxFastZerosIsDeprecated
#define mxUnreference mxUnreferenceIsDeprecated
#define mxUnshareArray mxUnshareArrayIsDeprecated
#define mxGetPropertyShared mxGetPropertySharedIsDeprecated
#define mxSetPropertyShared mxSetPropertySharedIsDeprecated

#endif /* TARGET_API_VERSION */

/*
 * allocate memory, notifying registered listener
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void *mxMalloc(size_t n /* number of bytes */
                                                   );

/*
 * allocate cleared memory, notifying registered listener.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void *mxCalloc(size_t n,   /* number of objects */
                                                   size_t size /* size of objects */
                                                   );

/*
 * free memory, notifying registered listener.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxFree(void *ptr) /* pointer to memory to be freed */;

/*
 * reallocate memory, notifying registered listener.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void *mxRealloc(void *ptr, size_t size);

/*
 * Get number of dimensions in array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mwSize mxGetNumberOfDimensions(const mxArray *pa);

/*
 * Get pointer to dimension array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C const mwSize *mxGetDimensions(const mxArray *pa);

/*
 * Get row dimension
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C size_t mxGetM(const mxArray *pa);

/*
 * Get row data pointer for sparse numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mwIndex *mxGetIr(const mxArray *pa);

/*
 * Get column data pointer for sparse numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mwIndex *mxGetJc(const mxArray *pa);

/*
 * Get maximum nonzero elements for sparse numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mwSize mxGetNzmax(const mxArray *pa);

/*
 * Set maximum nonzero elements for numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetNzmax(mxArray *pa, mwSize nzmax);

/*
 * Return pointer to the nth field name
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C const char *mxGetFieldNameByNumber(const mxArray *pa, int n);


/*
 * Return a pointer to the contents of the named field for
 * the ith element (zero based).
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxGetFieldByNumber(const mxArray *pa, mwIndex i, int fieldnum);

/*
 * Get a pointer to the specified cell element.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxGetCell(const mxArray *pa, mwIndex i);

/*
 * Return the class (category) of data that the array holds.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxClassID mxGetClassID(const mxArray *pa);

/*
 * Get pointer to data
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void *mxGetData(const mxArray *pa /* pointer to array */
                                                    );

/*
 * Set pointer to data
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetData(mxArray *pa,  /* pointer to array */
                                                   void *newdata /* pointer to data */
                                                   );

/*
 * Determine whether the specified array contains numeric (as opposed
 * to cell or struct) data.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsNumeric(const mxArray *pa);

/*
 * Determine whether the given array is a cell array.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsCell(const mxArray *pa);

/*
 * Determine whether the given array's logical flag is on.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsLogical(const mxArray *pa);

/*
 * Determine whether the given array's scalar flag is on.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsScalar(const mxArray *pa);

/*
 * Determine whether the given array contains character data.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsChar(const mxArray *pa);

/*
 * Determine whether the given array is a structure array.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsStruct(const mxArray *pa);

/*
 * Determine whether the given array is an opaque array.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsOpaque(const mxArray *pa);

/*
 * Returns true if specified array is a function object.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsFunctionHandle(const mxArray *pa);

/*
 * This function is deprecated and is preserved only for backward compatibility.
 * DO NOT USE if possible.
 * Is array user defined MATLAB v5 object
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsObject(const mxArray *pa /* pointer to array */
                                                    );
#if !defined(TARGET_API_VERSION) || (TARGET_API_VERSION == 700)
/*
 * Get imaginary data pointer for numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void *mxGetImagData(const mxArray *pa /* pointer to array */
                                                        );

/*
 * Set imaginary data pointer for numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void
mxSetImagData(mxArray *pa,  /* pointer to array */
              void *newdata /* imaginary data array pointer */
              );
#endif

/*
 * Determine whether the given array contains complex data.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsComplex(const mxArray *pa);

/*
 * Determine whether the given array is a sparse (as opposed to full).
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsSparse(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * double-precision floating-point numbers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsDouble(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * single-precision floating-point numbers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsSingle(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * signed 8-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsInt8(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * unsigned 8-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsUint8(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * signed 16-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsInt16(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * unsigned 16-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsUint16(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * signed 32-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsInt32(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * unsigned 32-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsUint32(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * signed 64-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsInt64(const mxArray *pa);

/*
 * Determine whether the specified array represents its data as
 * unsigned 64-bit integers.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsUint64(const mxArray *pa);

/*
 * Get number of elements in array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C size_t mxGetNumberOfElements(
    const mxArray *pa /* pointer to array */
    );

#if !defined(TARGET_API_VERSION) || (TARGET_API_VERSION == 700)
/*
 * Get imaginary data pointer for numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C double *mxGetPi(const mxArray *pa /* pointer to array */
                                                    );

/*
 * Set imaginary data pointer for numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetPi(mxArray *pa, /* pointer to array */
                                                 double *pi   /* imaginary data array pointer */
                                                 );
#endif

/*
 * Get string array data
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxChar *mxGetChars(const mxArray *pa /* pointer to array */
                                                       );

/*
 * Get 8 bits of user data stored in the mxArray header.  NOTE: The state
 * of these bits is not guaranteed to be preserved after API function
 * calls.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxGetUserBits(const mxArray *pa /* pointer to array */
                                                      );

/*
 * Set 8 bits of user data stored in the mxArray header. NOTE: The state
 * of these bits is not guaranteed to be preserved after API function
 * calls.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetUserBits(mxArray *pa, /* pointer to array */
                                                       int value);

/*
 * Get the real component of the specified array's first data element.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C double mxGetScalar(const mxArray *pa);

/*
 * Inform Watcom compilers that scalar double return values
 * will be in the FPU register.
 */
#ifdef __WATCOMC__
#pragma aux mxGetScalar value[8087];
#endif

/*
 * Is the isFromGlobalWorkspace bit set?
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsFromGlobalWS(const mxArray *pa);

/*
 * Set the isFromGlobalWorkspace bit.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetFromGlobalWS(mxArray *pa, bool global);

/*
 * Set row dimension
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetM(mxArray *pa, mwSize m);

/*
 * Get column dimension
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C size_t mxGetN(const mxArray *pa);

/*
 * Is array empty
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsEmpty(const mxArray *pa /* pointer to array */
                                                   );
/*
 * Get the index to the named field.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxGetFieldNumber(const mxArray *pa, const char *name);

/*
 * Set row data pointer for sparse numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetIr(mxArray *pa, mwIndex *newir);

/*
 * Set column data pointer for sparse numeric array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetJc(mxArray *pa, mwIndex *newjc);

LIBMMWMATRIX_PUBLISHED_API_EXTERN_C double *mxGetPr(const mxArray *pa);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetPr(mxArray *pa, double *newdata);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C size_t mxGetElementSize(const mxArray *pa);

/*
 * Return the offset (in number of elements) from the beginning of
 * the array to a given subscript.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mwIndex mxCalcSingleSubscript(const mxArray *pa,
                                                                     mwSize nsubs,
                                                                     const mwIndex *subs);

/*
 * Get number of structure fields in array
 * Returns 0 if mxArray is non-struct.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxGetNumberOfFields(const mxArray *pa /* pointer to array */
                                                            );

/*
 * Set an element in a cell array to the specified value.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetCell(mxArray *pa, mwIndex i, mxArray *value);

/*
 * Set pa[i][fieldnum] = value
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void 
mxSetFieldByNumber(mxArray *pa, mwIndex i, int fieldnum, mxArray *value);

/*
 * Return a pointer to the contents of the named field for the ith
 * element (zero based).  Returns NULL on no such field or if the
 * field itself is NULL
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxGetField(const mxArray *pa, mwIndex i, const char *fieldname);

/*
 * Sets the contents of the named field for the ith element (zero based).
 * The input 'value' is stored in the input array 'pa' - no copy is made.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void
mxSetField(mxArray *pa, mwIndex i, const char *fieldname, mxArray *value);

/*
 * mxGetProperty returns the value of a property for a given object and index.
 * The property must be public.
 *
 * If the given property name doesn't exist, or isn't public, or the object isn't
 * the right type, then mxGetProperty returns NULL.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxGetProperty(const mxArray *pa, const mwIndex i, const char *propname);

/*
 * mxSetProperty sets the value of a property for a given object and index.
 * The property must be public.
 *
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void
mxSetProperty(mxArray *pa, mwIndex i, const char *propname, const mxArray *value);

/*
 * Return the name of an array's class.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C const char *mxGetClassName(const mxArray *pa);

/*
 * Determine whether an array is a member of the specified class.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsClass(const mxArray *pa, const char *name);

/*
 * Create a numeric matrix and initialize all its data elements to 0.
 * In standalone mode, out-of-memory will mean a NULL pointer is returned.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateNumericMatrix(mwSize m, mwSize n, mxClassID classid, mxComplexity flag);

/*
 * Create an uninitialized numeric matrix.
 * The resulting array must be freed with mxDestroyArray.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateUninitNumericMatrix(size_t m, size_t n, mxClassID classid, mxComplexity flag);

/*
 * Create an uninitialized numeric array.
 * The resulting array must be freed with mxDestroyArray.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateUninitNumericArray(size_t ndim, size_t *dims, mxClassID classid, mxComplexity flag);

/*
 * Set column dimension
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetN(mxArray *pa, mwSize n);

/*
 * Set dimension array and number of dimensions.  Returns 0 on success and 1
 * if there was not enough memory available to reallocate the dimensions array.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int
mxSetDimensions(mxArray *pa, const mwSize *pdims, mwSize ndims);

/*
 * mxArray destructor
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxDestroyArray(mxArray *pa);

/*
 * Create a numeric array and initialize all its data elements to 0.
 *
 * As with mxCreateNumericMatrix, in a standalone application,
 * out-of-memory will mean a NULL pointer is returned.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateNumericArray(mwSize ndim, const mwSize *dims, mxClassID classid, mxComplexity flag);

/*
 * Create an N-Dimensional array to hold string data;
 * initialize all elements to 0.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateCharArray(mwSize ndim, const mwSize *dims);

/*
 * Create a two-dimensional array to hold double-precision
 * floating-point data; initialize each data element to 0.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateDoubleMatrix(mwSize m, mwSize n, mxComplexity flag);

/*
 * Get a properly typed pointer to the elements of a logical array.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxLogical *mxGetLogicals(const mxArray *pa);

/*
 * Create a logical array and initialize its data elements to false.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateLogicalArray(mwSize ndim, const mwSize *dims);

/*
 * Create a two-dimensional array to hold logical data and
 * initialize each data element to false.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateLogicalMatrix(mwSize m, mwSize n);

/*
 * Create a logical scalar mxArray having the specified value.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateLogicalScalar(bool value);

/*
 * Returns true if we have a valid logical scalar mxArray.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsLogicalScalar(const mxArray *pa);

/*
 * Returns true if the logical scalar value is true.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsLogicalScalarTrue(const mxArray *pa);

/*
 * Create a double-precision scalar mxArray initialized to the
 * value specified
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateDoubleScalar(double value);

/*
 * Create a 2-Dimensional sparse array.
 *
 * Z = mxCreateSparse(m,n,nzmax,cmplx_flag);
 * An m-by-n, real or complex, sparse matrix with room for nzmax nonzeros.
 * Use this to allocate storage for a sparse matrix.
 * It allocates the structure, allocates the pr, pi, ir and jc arrays,
 * and sets all the fields, which may be changed later.
 * Avoids the question of malloc(0) by replacing nzmax = 0 with nzmax = 1.
 * Also sets Z->pr[0] = 0.0 so that the scalar sparse(0.0) acts like 0.0.
 *
 * Notice that the value of m is almost irrelevant.  It is only stored in
 * the mxSetM field of the matrix structure.  It does not affect the amount
 * of storage required by sparse matrices, or the amount of time required
 * by sparse algorithms.  Consequently, m can be "infinite".  The result
 * is a semi-infinite matrix with a finite number of columns and a finite,
 * but unspecified, number of nonzero rows.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateSparse(mwSize m, mwSize n, mwSize nzmax, mxComplexity flag);

/*
 * Create a 2-D sparse logical array
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateSparseLogicalMatrix(mwSize m, mwSize n, mwSize nzmax);

/*
 * Copies characters from a MATLAB array to a char array
 * This function will attempt to perform null termination if it is possible.
 * nChars is the number of bytes in the output buffer
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void
mxGetNChars(const mxArray *pa, char *buf, mwSize nChars);

/*
 * Converts a string array to a C-style string. The C-style string is in the
 * local codepage encoding. If the conversion for the entire Unicode string
 * cannot fit into the supplied character buffer, then the conversion includes
 * the last whole codepoint that will fit into the buffer. The string is thus
 * truncated at the greatest possible whole codepoint and does not split code-
 * points.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int
mxGetString(const mxArray *pa, char *buf, mwSize buflen);

/*
 * Create a NULL terminated C string from an mxArray of type mxCHAR_CLASS
 * Supports multibyte character sets.  The resulting string must be freed
 * with mxFree.  Returns NULL on out of memory or non-character arrays.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C char *mxArrayToString(const mxArray *pa);

/*
 * Create a NULL terminated C string from an mxArray of type mxCHAR_CLASS
 * The C style string is in UTF-8 encoding. The resulting
 * string must be freed with mxFree. Returns NULL on out of memory or
 * non-character arrays.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C char *mxArrayToUTF8String(mxArray const *pa);

/**
 * Create a character vector initialized from the first n bytes in str. The
 * supplied string is presumed to be in the local codepage encoding. The
 * character data format in the mxArray will be UTF-16.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateStringFromNChars(const char *str, mwSize n);

/*
 * Create a character vector initialized from null-terminated string str. The
 * supplied string can use either UTF-8 encoding or the local codepage encoding.
 * The character data format in the mxArray will be UTF-16.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateString(const char *str);

/*
 * Create a string array initialized to the strings in str.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateCharMatrixFromStrings(mwSize m, const char **str);

/*
 * Create a 2-Dimensional cell array, with each cell initialized
 * to NULL.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateCellMatrix(mwSize m, mwSize n);

/*
 * Create an N-Dimensional cell array, with each cell initialized
 * to NULL.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateCellArray(mwSize ndim, const mwSize *dims);

/*
 * Create a 2-Dimensional structure array having the specified fields;
 * initialize all values to NULL.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateStructMatrix(mwSize m, mwSize n, int nfields, const char **fieldnames);

/*
 * Create an N-Dimensional structure array having the specified fields;
 * initialize all values to NULL.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *
mxCreateStructArray(mwSize ndim, const mwSize *dims, int nfields, const char **fieldnames);

/*
 * Make a deep copy of an array, return a pointer to the copy.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxDuplicateArray(const mxArray *in);

/*
 * Set classname of an unvalidated object array.  It is illegal to
 * call this function on a previously validated object array.
 * Return 0 for success, 1 for failure.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxSetClassName(mxArray *pa, const char *classname);

/*
 * Add a field to a structure array. Returns field number on success or -1
 * if inputs are invalid or an out of memory condition occurs.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxAddField(mxArray *pa, const char *fieldname);

/*
 * Remove a field from a structure array.  Does nothing if no such field exists.
 * Does not destroy the field itself.
*/
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxRemoveField(mxArray *pa, int field);

/*
 * Function for obtaining MATLAB's concept of EPS
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C double mxGetEps(void);

/*
 * Function for obtaining MATLAB's concept of INF (Used in MEX-File callback).
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C double mxGetInf(void);

/*
 * Function for obtaining MATLAB's concept of NaN (Used in MEX-File callback).
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C double mxGetNaN(void);

/*
 * test for finiteness in a machine-independent manner
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsFinite(double x /* value to test */
                                                    );

/*
 * test for infinity in a machine-independent manner
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsInf(double x /* value to test */
                                                 );
/*
 * test for NaN in a machine-independent manner
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C bool mxIsNaN(double x /* value to test */
                                                 );
#if !defined(TARGET_API_VERSION) || (TARGET_API_VERSION == 700)
/*
 * Undocumented.  Removed in later APIs.
 */
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateSharedDataCopy(const mxArray *pa);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxCreateUninitDoubleMatrix(int cmplx_flag, size_t m, size_t n);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxFastZeros(int cmplx_flag, int m, int n);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxUnreference(mxArray *pa);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxUnshareArray(mxArray *pa, int level);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mxArray *mxGetPropertyShared(const mxArray *pa, size_t i, const char *propname);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void mxSetPropertyShared(mxArray *pa, size_t i, const char *propname, const mxArray *value);
#endif


#if TARGET_API_VERSION >= 800

/*
 * Typed data access for numeric arrays
 */
#define MX_DECLARE_DATA_ACCESSORS(Name)                                                              \
    LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mx##Name *mxGet##Name##s(mxArray const *);                   \
    LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxSet##Name##s(mxArray *, mx##Name *);                   \
    LIBMMWMATRIX_PUBLISHED_API_EXTERN_C mx##Complex##Name *mxGet##Complex##Name##s(mxArray const *); \
    LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxSet##Complex##Name##s(mxArray *, mx##Complex##Name *)

MX_DECLARE_DATA_ACCESSORS(Double); /* mxDoubles*, mxComplexDoubles* in mx[SG]etDoubles, mx[SG]etComplexDoubles */
MX_DECLARE_DATA_ACCESSORS(Single); /* mxSingles*, mxComplexSingles* in mx[SG]etSingles, mx[SG]etComplexSingles */
MX_DECLARE_DATA_ACCESSORS(Int8);   /* mxInt8s*,   mxComplexInt8s*   in mx[SG]etInt8s,   mx[SG]etComplexInt8s   */
MX_DECLARE_DATA_ACCESSORS(Uint8);  /* mxUint8s*,  mxComplexUint8s*  in mx[SG]etUint8s,  mx[SG]etComplexUint8s  */
MX_DECLARE_DATA_ACCESSORS(Int16);  /* mxInt16s*,  mxComplexInt16s*  in mx[SG]etInt16s,  mx[SG]etComplexInt16s  */
MX_DECLARE_DATA_ACCESSORS(Uint16); /* mxUint16s*, mxComplexUint16s* in mx[SG]etUint16s, mx[SG]etComplexUint16s */
MX_DECLARE_DATA_ACCESSORS(Int32);  /* mxInt32s*,  mxComplexInt32s*  in mx[SG]etInt32s,  mx[SG]etComplexInt32s  */
MX_DECLARE_DATA_ACCESSORS(Uint32); /* mxUint32s*, mxComplexUint32s* in mx[SG]etUint32s, mx[SG]etComplexUint32s */
MX_DECLARE_DATA_ACCESSORS(Int64);  /* mxInt64s*,  mxComplexInt64s*  in mx[SG]etInt64s,  mx[SG]etComplexInt64s  */
MX_DECLARE_DATA_ACCESSORS(Uint64); /* mxUint64s*, mxComplexUint64s* in mx[SG]etUint64s, mx[SG]etComplexUint64s */

LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxMakeArrayReal(mxArray *);
LIBMMWMATRIX_PUBLISHED_API_EXTERN_C int mxMakeArrayComplex(mxArray *);

#endif /* TARGET_API_VERSION >= 800 */

/*
mxAssert(int expression, char *error_message)
---------------------------------------------

  Similar to ANSI C's assert() macro, the mxAssert macro checks the
  value of an assertion, continuing execution only if the assertion
  holds.  If 'expression' evaluates to be true, then the mxAssert does
  nothing.  If, however, 'expression' is false, then mxAssert prints an
  error message to the MATLAB Command Window, consisting of the failed
  assertion's expression, the file name and line number where the failed
  assertion occurred, and the string 'error_message'.  'error_message'
  allows the user to specify a more understandable description of why
  the assertion failed.  (Use an empty string if no extra description
  should follow the failed assertion message.)  After a failed
  assertion, control returns to the MATLAB command line.

  mxAssertS, (the S for Simple), takes the same inputs as mxAssert.  It
  does not print the text of the failed assertion, only the file and
  line where the assertion failed, and the explanatory error_message.

  Note that script MEX will turn off these assertions when building
  optimized MEX-functions, so they should be used for debugging
  purposes only.
*/

#ifdef MATLAB_MEX_FILE
#ifndef NDEBUG

LIBMMWMATRIX_PUBLISHED_API_EXTERN_C void
mexPrintAssertion(const char *test, const char *fname, int linenum, const char *message);

#define mxAssert(test, message)                                                                    \
    ((test) ? (void)0 : mexPrintAssertion(#test, __FILE__, __LINE__, message))
#define mxAssertS(test, message)                                                                   \
    ((test) ? (void)0 : mexPrintAssertion("", __FILE__, __LINE__, message))
#else
#define mxAssert(test, message) ((void)0)
#define mxAssertS(test, message) ((void)0)
#endif
#else
#include <assert.h>
#define mxAssert(test, message) assert(test)
#define mxAssertS(test, message) assert(test)
#endif

/* Current MATRIX published API version */
#define MX_CURRENT_API_VER 0x08000000

/* Backward compatible current MATRIX published API version */
#define MX_API_VER MX_CURRENT_API_VER

/* Backward compatible MATRIX published API versions */
#define MX_LAST_32BIT_VER 0x07000000
#define MX_LAST_SEPARATE_COMPLEX_VER 0x07300000

/* Required MEX-file MATRIX published API version */
#if TARGET_API_VERSION == 700
#if defined(MX_COMPAT_32)
#define MX_TARGET_API_VER MX_LAST_32BIT_VER
#else
#define MX_TARGET_API_VER MX_LAST_SEPARATE_COMPLEX_VER
#endif
#else
#define MX_TARGET_API_VER MX_CURRENT_API_VER
#endif

/*
 * The following macros enable conditional compilation based on the
 * target published API. The macros can be used in a single source file
 * that is intended to be built against multiple matrix API versions.
 *
 * MX_HAS_64BIT_ARRAY_DIMS evaluates to a non-zero value if array
 * dimensions are 64 bits wide.
 *
 * MX_HAS_INTERLEAVED_COMPLEX evaluates to a non-zero value if complex
 * array data is interleaved.
 *
 */
#define MX_HAS_64BIT_ARRAY_DIMS MX_TARGET_API_VER > MX_LAST_32BIT_VER
#define MX_HAS_INTERLEAVED_COMPLEX MX_TARGET_API_VER > MX_LAST_SEPARATE_COMPLEX_VER

/*
 * Inform Watcom compilers that scalar double return values
 * will be in the FPU register.
 */
#ifdef __WATCOMC__
#pragma aux mxGetEps value[8087];
#pragma aux mxGetInf value[8087];
#pragma aux mxGetNaN value[8087];
#endif

#endif /* matrix_h */
