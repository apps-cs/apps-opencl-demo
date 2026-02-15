/** *************************************************************************
 *
 * @internal
 *   Demo program for teaching the course 
 *   Computer Architectures and Parallel Systems.
 *
 *   GPU Programming using OpenCL
 *
 *   02/2026, Petr Olivka, Dep. of Computer Science, FEI, VSB-TU Ostrava
 *   petr.olivka@vsb.cz
 *   https:/poli.cs.vsb.cz/edu/apps
 * @endinternal
 *
 * @file ocl_svm_mat_allocator.h
 * @brief Share Virtual Memory Mat Allocator
 *
 * @details
 * Header file for cv::Mat Allocator class using Share Virtual Memory (SVM).
 * 
 ***************************************************************************/

#ifndef __OCL_SVM_MAT_ALLOCATOR
#define __OCL_SVM_MAT_ALLOCATOR

#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>

/**
 * @brief Class for cv::Mat Allocator using Share Virtual Memory (SVM).
 *
 * Share Virtual Memory allocator for cv::Mat class. 
 * SVMMatAllocator was created using StdMatAllocator, part of OpenCV project. 
 * See https://github.com/opencv/opencv/blob/4.x/modules/core/src/matrix.cpp.
*/

class SVMMatAllocator : public cv::MatAllocator
{
public:

/**
 * @brief Data Allocator
 * @param dims Number of dimensions.
 * @param sizez Individual dimensions.
 * @param type Data type CV_...
 * @param data0 Externally allocated data.
 * @param step Number of bytes between individual dimensions.
 * @param cv::AccessFlag ACCESS_..., see OpenCV.
 * @param cv::UMatUsageFlag USAGE_..., see OpenCV.
 * @return *UMatData object.
*/
    cv::UMatData* allocate(int dims, const int* sizes, int type,
                       void* data0, size_t* step, cv::AccessFlag /*flags*/, cv::UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE;

/**
 * @brief Verification of memory availability. 
 * @param cv::UmatData Existing cv::Mat object.
 * @param cv::AccessFlag ACCESS_..., see OpenCV.
 * @param cv::UMatUsageFlag USAGE_..., see OpenCV.
 * @return true - memory is prepared / false - allocation failed
*/
    bool allocate(cv::UMatData* u, cv::AccessFlag /*accessFlags*/, cv::UMatUsageFlags /*usageFlags*/) const CV_OVERRIDE;

/**
 * @brief Data Deallocator
 * @param cv::UMatData Allocated object.
*/
    void deallocate(cv::UMatData* u) const CV_OVERRIDE;
};

#endif // __OCL_SVM_MAT_ALLOCATOR
       
