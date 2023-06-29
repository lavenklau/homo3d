#include "templateMatrix.h"
//#include "snippet.h"
#include "utils.h"
#include "math.h"
#include "matlab/matlab_utils.h"

// Kelam * 72
char kelam72[] = {
	8,6,6,-8,6,6,4,-6,3,-4,-6,3,4,3,-6,-4,3,-6,2,-3,-3,-2,-3,-3,
	6,8,6,-6,4,3,6,-8,6,-6,-4,3,3,4,-6,-3,2,-3,3,-4,-6,-3,-2,-3,
	6,6,8,-6,3,4,3,-6,4,-3,-3,2,6,6,-8,-6,3,-4,3,-6,-4,-3,-3,-2,
	-8,-6,-6,8,-6,-6,-4,6,-3,4,6,-3,-4,-3,6,4,-3,6,-2,3,3,2,3,3,
	6,4,3,-6,8,6,6,-4,3,-6,-8,6,3,2,-3,-3,4,-6,3,-2,-3,-3,-4,-6,
	6,3,4,-6,6,8,3,-3,2,-3,-6,4,6,3,-4,-6,6,-8,3,-3,-2,-3,-6,-4,
	4,6,3,-4,6,3,8,-6,6,-8,-6,6,2,3,-3,-2,3,-3,4,-3,-6,-4,-3,-6,
	-6,-8,-6,6,-4,-3,-6,8,-6,6,4,-3,-3,-4,6,3,-2,3,-3,4,6,3,2,3,
	3,6,4,-3,3,2,6,-6,8,-6,-3,4,3,6,-4,-3,3,-2,6,-6,-8,-6,-3,-4,
	-4,-6,-3,4,-6,-3,-8,6,-6,8,6,-6,-2,-3,3,2,-3,3,-4,3,6,4,3,6,
	-6,-4,-3,6,-8,-6,-6,4,-3,6,8,-6,-3,-2,3,3,-4,6,-3,2,3,3,4,6,
	3,3,2,-3,6,4,6,-3,4,-6,-6,8,3,3,-2,-3,6,-4,6,-3,-4,-6,-6,-8,
	4,3,6,-4,3,6,2,-3,3,-2,-3,3,8,6,-6,-8,6,-6,4,-6,-3,-4,-6,-3,
	3,4,6,-3,2,3,3,-4,6,-3,-2,3,6,8,-6,-6,4,-3,6,-8,-6,-6,-4,-3,
	-6,-6,-8,6,-3,-4,-3,6,-4,3,3,-2,-6,-6,8,6,-3,4,-3,6,4,3,3,2,
	-4,-3,-6,4,-3,-6,-2,3,-3,2,3,-3,-8,-6,6,8,-6,6,-4,6,3,4,6,3,
	3,2,3,-3,4,6,3,-2,3,-3,-4,6,6,4,-3,-6,8,-6,6,-4,-3,-6,-8,-6,
	-6,-3,-4,6,-6,-8,-3,3,-2,3,6,-4,-6,-3,4,6,-6,8,-3,3,2,3,6,4,
	2,3,3,-2,3,3,4,-3,6,-4,-3,6,4,6,-3,-4,6,-3,8,-6,-6,-8,-6,-6,
	-3,-4,-6,3,-2,-3,-3,4,-6,3,2,-3,-6,-8,6,6,-4,3,-6,8,6,6,4,3,
	-3,-6,-4,3,-3,-2,-6,6,-8,6,3,-4,-3,-6,4,3,-3,2,-6,6,8,6,3,4,
	-2,-3,-3,2,-3,-3,-4,3,-6,4,3,-6,-4,-6,3,4,-6,3,-8,6,6,8,6,6,
	-3,-2,-3,3,-4,-6,-3,2,-3,3,4,-6,-6,-4,3,6,-8,6,-6,4,3,6,8,6,
	-3,-3,-2,3,-6,-4,-6,3,-4,6,6,-8,-3,-3,2,3,-6,4,-6,3,4,6,6,8
};

// keMu * 72
char kemu72[] = {
	32,6,6,-8,-6,-6,4,6,3,-10,-6,-3,4,3,6,-10,-3,-6,-4,3,3,-8,-3,-3,
	6,32,6,6,4,3,-6,-8,-6,-6,-10,-3,3,4,6,3,-4,3,-3,-10,-6,-3,-8,-3,
	6,6,32,6,3,4,3,6,4,3,3,-4,-6,-6,-8,-6,-3,-10,-3,-6,-10,-3,-3,-8,
	-8,6,6,32,-6,-6,-10,6,3,4,-6,-3,-10,3,6,4,-3,-6,-8,3,3,-4,-3,-3,
	-6,4,3,-6,32,6,6,-10,-3,6,-8,-6,-3,-4,3,-3,4,6,3,-8,-3,3,-10,-6,
	-6,3,4,-6,6,32,-3,3,-4,-3,6,4,6,-3,-10,6,-6,-8,3,-3,-8,3,-6,-10,
	4,-6,3,-10,6,-3,32,-6,6,-8,6,-6,-4,-3,3,-8,3,-3,4,-3,6,-10,3,-6,
	6,-8,6,6,-10,3,-6,32,-6,-6,4,-3,3,-10,6,3,-8,3,-3,4,-6,-3,-4,-3,
	3,-6,4,3,-3,-4,6,-6,32,6,-3,4,-3,6,-10,-3,3,-8,-6,6,-8,-6,3,-10,
	-10,-6,3,4,6,-3,-8,-6,6,32,6,-6,-8,-3,3,-4,3,-3,-10,-3,6,4,3,-6,
	-6,-10,3,-6,-8,6,6,4,-3,6,32,-6,-3,-8,3,-3,-10,6,3,-4,-3,3,4,-6,
	-3,-3,-4,-3,-6,4,-6,-3,4,-6,-6,32,3,3,-8,3,6,-10,6,3,-10,6,6,-8,
	4,3,-6,-10,-3,6,-4,3,-3,-8,-3,3,32,6,-6,-8,-6,6,4,6,-3,-10,-6,3,
	3,4,-6,3,-4,-3,-3,-10,6,-3,-8,3,6,32,-6,6,4,-3,-6,-8,6,-6,-10,3,
	6,6,-8,6,3,-10,3,6,-10,3,3,-8,-6,-6,32,-6,-3,4,-3,-6,4,-3,-3,-4,
	-10,3,-6,4,-3,6,-8,3,-3,-4,-3,3,-8,6,-6,32,-6,6,-10,6,-3,4,-6,3,
	-3,-4,-3,-3,4,-6,3,-8,3,3,-10,6,-6,4,-3,-6,32,-6,6,-10,3,6,-8,6,
	-6,3,-10,-6,6,-8,-3,3,-8,-3,6,-10,6,-3,4,6,-6,32,3,-3,-4,3,-6,4,
	-4,-3,-3,-8,3,3,4,-3,-6,-10,3,6,4,-6,-3,-10,6,3,32,-6,-6,-8,6,6,
	3,-10,-6,3,-8,-3,-3,4,6,-3,-4,3,6,-8,-6,6,-10,-3,-6,32,6,-6,4,3,
	3,-6,-10,3,-3,-8,6,-6,-8,6,-3,-10,-3,6,4,-3,3,-4,-6,6,32,-6,3,4,
	-8,-3,-3,-4,3,3,-10,-3,-6,4,3,6,-10,-6,-3,4,6,3,-8,-6,-6,32,6,6,
	-3,-8,-3,-3,-10,-6,3,-4,3,3,4,6,-6,-10,-3,-6,-8,-6,6,4,3,6,32,6,
	-3,-3,-8,-3,-6,-10,-6,-3,-10,-6,-6,-8,3,3,-4,3,6,4,6,3,4,6,6,32
};

double LamConst, MuConst;

double LaMuSetFull[14];
double LaMuSetSimple[5];

using namespace homo;

/* initialize elastic matrix for further computation */
Scalar mu = default_poisson_ratio;
Scalar E = default_youngs_modulus;
Eigen::Matrix<double, 6, 6> elastic_matrix;
Eigen::Matrix<double, 24, 24> Ke;
Eigen::Matrix<float, 24, 24> fKe;

Eigen::Matrix<double, 8, 8> KT;
Eigen::Matrix<float, 8, 8> fKT;

Scalar* g_Ke;

Eigen::Matrix<double, 3, 1> dN(int i, double elen, Eigen::Matrix<double, 3, 1>& param) {
	if (i > 7 || i < 0) throw std::runtime_error("");
	int id[3] = { (i % 2),   (i / 2 % 2),   (i / 4) };
	double r[3];
	for (int k = 0; k < 3; k++) {
		r[k] = (id[k] ? param[k] : (1 - param[k]));
	}

	Eigen::Matrix<double, 3, 1> dn;
	for (int k = 0; k < 3; k++) {
		homoutils::Zp<3> j;
		dn[k] = (id[k] ? 1.0 / elen : -1.0 / elen) * r[j[k + 1]] * r[j[k + 2]];
	}
	return dn;
}

void initTemplateMatrix(
	Scalar element_len, BufferManager& gm, Scalar ymodu /*= default_youngs_modulus*/, Scalar ps_ratio /*= default_poisson_ratio*/)
{
	mu = ps_ratio;
	E = ymodu;
	elastic_matrix << 1 - mu, mu, mu, 0, 0, 0,
		mu, 1 - mu, mu, 0, 0, 0,
		mu, mu, 1 - mu, 0, 0, 0,
		0, 0, 0, (1 - 2 * mu) / 2, 0, 0,
		0, 0, 0, 0, (1 - 2 * mu) / 2, 0,
		0, 0, 0, 0, 0, (1 - 2 * mu) / 2;
	elastic_matrix *= E / ((1 + mu)*(1 - 2 * mu));

	Eigen::Matrix<double, 3, 1> gs_points[8];
	
	double p = sqrt(1. / 3);

	for (int i = 0; i < 8; i++) {
		int x = 2 * (i % 2) - 1;
		int y = 2 * (i / 2 % 2) - 1;
		int z = 2 * (i / 4) - 1;
		gs_points[i][0] = (x * p + 1) / 2;
		gs_points[i][1] = (y * p + 1) / 2;
		gs_points[i][2] = (z * p + 1) / 2;
	}

	
	Ke.fill(0);
	// Gauss Quadrature Point
	for (int i = 0; i < 8; i++) {

		Eigen::Matrix<double, 3, 1> grad_N[8];

		// Element Vertex Point
		for (int k = 0; k < 8; k++) {
			grad_N[k] = dN(k, element_len, gs_points[i]);
		}

		Eigen::Matrix<double, 6, 24> B;

		B.fill(0);

		for (int a = 0; a < 3; a++) {
			int offset = a;
			for (int b = 0; b < 8; b++) {
				B(a, offset) = grad_N[b][a];
				offset += 3;
			}
		}
		int offset = 0;
		/// torsional strain tau
		for (int b = 0; b < 8; b++) {
			/// tau_yz
			B(3, offset + 1) = grad_N[b].z();
			B(3, offset + 2) = grad_N[b].y();
			/// tau_xz
			B(4, offset) = grad_N[b].z();
			B(4, offset + 2) = grad_N[b].x();
			/// tau_xy
			B(5, offset) = grad_N[b].y();
			B(5, offset + 1) = grad_N[b].x();

			offset += 3;
		}
		Ke += B.transpose() * elastic_matrix * B;
	}

	Ke *= pow(element_len / 2, 3);

	Ke = (Ke + Ke.transpose()) / 2;

	fKe = Ke.cast<float>();

	eigen2ConnectedMatlab("KE", Ke);
	//g_Ke = (double*)gm.add_buf("template matrix buf ", sizeof(Ke), Ke.data());

	// DEBUG
	// compute rigid motion on element
	Eigen::Matrix<double, 24, 6> RE;
	for (int i = 0; i < 8; i++) {
		RE.block<3, 3>(i * 3, 0) = Eigen::Matrix<double, 3, 3>::Identity();
		Eigen::Matrix3d phat;
		int p[3] = { i % 2, i / 2 % 2, i / 2 / 2 };
		phat << 0, -p[2], p[1],
			p[2], 0, -p[0],
			-p[1], p[0], 0;
		RE.block<3, 3>(i * 3, 3) = phat;
	}
	for (int i = 0; i < 6; i++) {
		for (int j = 0; j < i; j++) {
			RE.col(i) -= RE.col(i).dot(RE.col(j)) * RE.col(j);
		}
		RE.col(i).normalize();
	}

	//eigen2ConnectedMatlab("RE", RE);

	auto E = ymodu;
	auto v = ps_ratio;
	LamConst = v * E / ((1 + v) * (1 - 2 * v));
	MuConst = E / (2 * (1 + v));
	LaMuSetSimple[0] = (LamConst - 2 * MuConst) / 72;
	LaMuSetSimple[1] = (LamConst - MuConst) / 72;
	LaMuSetSimple[2] = (LamConst + MuConst) / 72;
	LaMuSetSimple[3] = (LamConst + 4 * MuConst) / 72;
	LaMuSetSimple[4] = (2 * LamConst + 5 * MuConst) / 72;
}

const Eigen::Matrix<Scalar, 24, 24>& getTemplateMatrix(void)
{
	return fKe;
}

const Eigen::Matrix<Scalar, 8, 8> &getHeatTemplateMatrix(void)
{
	return fKT;
}

const Eigen::Matrix<double, 8, 8>& getHeatTemplateMatrixFp64(void) {
	return KT;
}

const Eigen::Matrix<double, 24, 24>& getTemplateMatrixFp64(void)
{
	return Ke;
}

const Scalar* getTemplateMatrixElements(void)
{
	return fKe.data();
}

Scalar* getDeviceTemplateMatrix(void) {
	return g_Ke;
}

double* getLamMuset(void) {
	return LaMuSetSimple;
}

const char* getKeLam72(void)
{
	return kelam72;
}

const char* getKeMu72(void)
{
	return kemu72;
}

template <int modulu = 0>
struct Zp
{
	int operator[](int n)
	{
		n -= modulu * (n / modulu);
		n += modulu;
		return n % modulu;
	}
};

Eigen::Matrix<Scalar, 3, 1> dN(int i, Scalar elen, Eigen::Matrix<Scalar, 3, 1>& param) {
	if (i > 7 || i < 0) throw std::runtime_error("");
	int id[3] = { (i % 2),   (i / 2 % 2),   (i / 4) };
	Scalar r[3];
	for (int k = 0; k < 3; k++) {
		r[k] = (id[k] ? param[k] : (1 - param[k]));
	}

	Eigen::Matrix<Scalar, 3, 1> dn;
	for (int k = 0; k < 3; k++) {
		Zp<3> j;
		dn[k] = (id[k] ? 1.0 / elen : -1.0 / elen) * r[j[k + 1]] * r[j[k + 2]];
	}
	return dn;
}

void initHeatTemplateMatrix(void) {
	Eigen::Matrix<Scalar, 3, 1> gs_points[8];
	double p = sqrt(3) / 3;
	for (int i = 0; i < 8; i++) {
		int x = 2 * (i % 2) - 1;
		int y = 2 * (i / 2 % 2) - 1;
		int z = 2 * (i / 4) - 1;
		gs_points[i][0] = (x * p + 1) / 2;
		gs_points[i][1] = (y * p + 1) / 2;
		gs_points[i][2] = (z * p + 1) / 2;
	}
	KT.fill(0);
	// Gauss Quadrature Point
	for (int i = 0; i < 8; i++) {
		Eigen::Matrix<Scalar, 3, 1> grad_N[8];
		// Element Vertex Point
		for (int k = 0; k < 8; k++) {
			grad_N[k] = dN(k, 1, gs_points[i]);
		}
		for (int row = 0; row < 8; row++) {
			for (int col = 0; col < 8; col++) {
				KT(row, col) += grad_N[row].dot(grad_N[col]);
			}
		}
	}
	std::cout << "KT = \n" << KT << std::endl;
	fKT = KT.cast<float>();
}