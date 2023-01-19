/*
 * Copyright 1984-2017 The MathWorks, Inc.
 */

#if defined(_MSC_VER)
# pragma once
#endif
#if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ > 3))
# pragma once
#endif

#ifndef PUBLISHED_EXTERN_API_HPP
#define PUBLISHED_EXTERN_API_HPP

#if defined(BUILDING_LIBENG)
 #define LIBMWENGINE_PUBLISHED_API DLL_EXPORT_SYM
#else
 #define LIBMWENGINE_PUBLISHED_API
#endif

#ifndef EXTERN_C
 #ifdef __cplusplus
  #define EXTERN_C extern "C"
 #else
  #define EXTERN_C extern
 #endif
#endif

#ifndef LIBMWENGINE_PUBLISHED_API_EXTERN_C
 #define LIBMWENGINE_PUBLISHED_API_EXTERN_C EXTERN_C LIBMWENGINE_PUBLISHED_API
#endif


#include "matrix.h"

typedef struct engine Engine;

/* Execute matlab statement */
LIBMWENGINE_PUBLISHED_API_EXTERN_C int engEvalString(Engine *ep,
  const char *string);

/* Start matlab process for single use.
   Not currently supported on UNIX. */
LIBMWENGINE_PUBLISHED_API_EXTERN_C Engine *engOpenSingleUse(
  const char *startcmd, void *reserved, int *retstatus);

/* SetVisible, do nothing since this function is only for NT  */
LIBMWENGINE_PUBLISHED_API_EXTERN_C int engSetVisible(Engine *ep, bool newVal);

/* GetVisible, do nothing since this function is only for NT */
LIBMWENGINE_PUBLISHED_API_EXTERN_C int engGetVisible(Engine *ep, bool* bVal);

/* Start matlab process */
LIBMWENGINE_PUBLISHED_API_EXTERN_C Engine *engOpen(const char *startcmd);

/* Start matlab process */
LIBMWENGINE_PUBLISHED_API_EXTERN_C Engine *fengOpen(const char *startcmd);

/* Close down matlab server */
LIBMWENGINE_PUBLISHED_API_EXTERN_C int engClose(Engine *ep);

/* Get a variable with the specified name from MATLAB's workspace */
LIBMWENGINE_PUBLISHED_API_EXTERN_C mxArray *engGetVariable(
  Engine *ep, const char *name);

/* Put a variable into MATLAB's workspace with the specified name */
LIBMWENGINE_PUBLISHED_API_EXTERN_C int engPutVariable(Engine *ep,
  const char *var_name, const mxArray *ap);

/* register a buffer to hold matlab text output */
LIBMWENGINE_PUBLISHED_API_EXTERN_C int engOutputBuffer(Engine *ep,
  char *buffer, int buflen);

#endif /* PUBLISHED_EXTERN_API_HPP */
