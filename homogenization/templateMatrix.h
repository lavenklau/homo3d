#pragma once

#ifndef __TEMPLATE_MATRIX_H
#define __TEMPLATE_MATRIX_H

#include "Eigen/Eigen"
#include "gmem/DeviceBuffer.h"

typedef float Scalar;

constexpr double default_poisson_ratio = 0.3;
constexpr double default_youngs_modulus = 1e6;


void initTemplateMatrix(Scalar element_len, homo::BufferManager& gm, Scalar ymodu = default_youngs_modulus, Scalar ps_ratio = default_poisson_ratio);

const Eigen::Matrix<Scalar, 24, 24>& getTemplateMatrix(void);

const Eigen::Matrix<double, 24, 24>& getTemplateMatrixFp64(void);

const Scalar* getTemplateMatrixElements(void);

Scalar* getDeviceTemplateMatrix(void);

const char* getKeLam72(void);

const char* getKeMu72(void);

#endif

