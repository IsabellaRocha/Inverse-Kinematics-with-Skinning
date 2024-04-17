#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
	int numIKJoints, const int* IKJointIDs, const FK& fk,
	const std::vector<real>& eulerAngles, std::vector<real>& handlePositions)
{
	// Students should implement this.
	// The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
	// The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
	// Then, implement the same algorithm into this function. To do so,
	// you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
	// Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
	// It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
	// so that code is only written once. We considered this; but it is actually not easily doable.
	// If you find a good approach, feel free to document it in the README file, for extra credit.

	vector<Mat3<real>> localTransformation(fk.getNumJoints());
	vector<Mat3<real>> globalTransformation(fk.getNumJoints());

	vector<Vec3<real>> localTranslation(fk.getNumJoints());
	vector<Vec3<real>> globalTranslation(fk.getNumJoints());

	// Similar to that in FK.cpp:

	for (int idx = 0; idx < fk.getNumJoints(); idx++) {
		real eulerArray[3];
		real orientationArray[3];

		for (int i = 0; i < 3; i++) {
			eulerArray[i] = eulerAngles[3 * idx + i];
			orientationArray[i] = fk.getJointOrient(idx).data()[i];
			localTranslation[idx][i] = fk.getJointRestTranslation(idx)[i];

			globalTranslation[fk.getJointUpdateOrder(idx)][i] = fk.getJointRestTranslation(fk.getJointUpdateOrder(idx))[i];
			globalTranslation[fk.getJointUpdateOrder(idx)][i] = fk.getJointRestTranslation(fk.getJointUpdateOrder(idx))[i];
			globalTranslation[fk.getJointUpdateOrder(idx)][i] = fk.getJointRestTranslation(fk.getJointUpdateOrder(idx))[i];
		}

		Mat3<real> eulerMatrix = Euler2Rotation(eulerArray, fk.getJointRotateOrder(idx));
		Mat3<real> orientationMatrix = Euler2Rotation(orientationArray, XYZ);

		localTransformation[idx] = orientationMatrix * eulerMatrix;

		// Check for root
		if (fk.getJointParent(fk.getJointUpdateOrder(idx)) == -1) {
			globalTransformation[fk.getJointUpdateOrder(idx)] = localTransformation[fk.getJointUpdateOrder(idx)];
		}
		else {
			//multiplyAffineTransform4ds(gTransformation[parent], gTranslation[parent], lTransformation[child], lTranslation[child], globalTransformation[child], globalTranslation[child])
			multiplyAffineTransform4ds(
				globalTransformation[fk.getJointParent(fk.getJointUpdateOrder(idx))],
				globalTranslation[fk.getJointParent(fk.getJointUpdateOrder(idx))],
				localTransformation[fk.getJointUpdateOrder(idx)],
				localTranslation[fk.getJointUpdateOrder(idx)],
				globalTransformation[fk.getJointUpdateOrder(idx)],
				globalTranslation[fk.getJointUpdateOrder(idx)]
			);
		}	
	}

	for (int idx = 0; idx < numIKJoints; idx++)
	{
		for (int i = 0; i < 3; i++) {
			handlePositions[3 * idx + i] = globalTranslation[IKJointIDs[idx]][i];
		}
	}

}

} // end anonymous namespaces


IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
	// Students should implement this.
	// Here, you should setup adol_c:
	//   Define adol_c inputs and outputs. 
	//   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
	//   This will later make it possible for you to compute the gradient of this function in IK::doIK
	//   (in other words, compute the "Jacobian matrix" J).
	// See ADOLCExample.cpp .
	
	int n = FKInputDim; // forward dynamics input dimension
	int m = FKOutputDim; // forward dynamics output dimension

	trace_on(adolc_tagID); //start tracking computation with ADOL-C

	vector<adouble> x(n); //define the input of function f
	for (int i = 0; i < n; i++) {
		x[i] <<= 0;
	}

	vector<adouble> y(m); //define the output of the function f
	forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, x, y);

	vector<double> output(m);
	for (int i = 0; i < m; i++) {
		y[i] >>= output[i];
	}

	// ADOL-C tracking finished
	trace_off();
}

void IK::doIK(const Vec3d* targetHandlePositions, Vec3d* jointEulerAngles)
{
	// You may find the following helpful:
	int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!

	// Students should implement this.
	// Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
	// Specifically, use ::function, and ::jacobian .
	// See ADOLCExample.cpp .
	//
	// Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
	// Note that at entry, "jointEulerAngles" contains the input Euler angles. 
	// Upon exit, jointEulerAngles should contain the new Euler angles.

	int n = FKInputDim; // forward dynamics input dimension
	int m = FKOutputDim; // forward dynamics output dimension

	// Set output handle positions
	vector<double> y(m);

	// data() to flatten out vec3d
	::function(adolc_tagID, m, n, jointEulerAngles->data(), y.data());

	vector<double> jacobianMatrix(m * n);
	vector<double*> jacobianMatrixEachRow(m);

	for (int idx = 0; idx < m; idx++) {
		jacobianMatrixEachRow[idx] = &jacobianMatrix[idx * n];
	}

	::jacobian(adolc_tagID, m, n, jointEulerAngles->data(), jacobianMatrixEachRow.data());

	Eigen::MatrixXd Jacobian(m, n);

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			Jacobian(i, j) = jacobianMatrix[i * n + j];
		}
	}
		
	vector<double> delta(m);
	for (int i = 0; i < m; i++)
	{
		delta[i] = targetHandlePositions->data()[i] - y.data()[i];
	}

	//Convert to VectorXd
	Eigen::Map<Eigen::VectorXd> deltaB(delta.data(), delta.size());

	double alpha = 0.001;
	Eigen::MatrixXd Identity = Eigen::MatrixXd::Identity(n, n);
	Eigen::MatrixXd JacobianTranspose = Jacobian.transpose();

	// Full equation: (Jt*J + alpha * I)deltaTheta = Jt * deltaB
	// Jt*J + alpha * I
	Eigen::MatrixXd LeftSideOfEquation(n, n);
	LeftSideOfEquation = JacobianTranspose * Jacobian + alpha * Identity;

	// Jt * deltaB
	Eigen::VectorXd RightSideOfEquation(m);
	RightSideOfEquation = JacobianTranspose * deltaB;

	// Solve for deltaTheta
	Eigen::VectorXd deltaTheta = LeftSideOfEquation.ldlt().solve(RightSideOfEquation);

	for (int i = 0; i < numJoints; i++) {
		for (int j = 0; j < 3; j++) {
			jointEulerAngles[i][j] = jointEulerAngles[i][j] + deltaTheta[3 * i + j];
		}
	}
}
