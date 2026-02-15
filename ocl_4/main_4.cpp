/** *************************************************************************
 *
 * Demo program for teaching the course
 * Computer Architectures and Parallel Systems.
 *
 * GPU Programming using OpenCL
 *
 * 02/2026, Petr Olivka, Dep. of Computer Science, FEI, VSB-TU Ostrava
 * petr.olivka@vsb.cz
 * https:/poli.cs.vsb.cz/edu/apps
 *
 * Converting BGR image to Black&White image.
 * Two images with different color depth are used. 
 *
 ***************************************************************************/

#include <cstdlib>
#include <ostream>
#include <unistd.h>
#include <iostream>
#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>

#include <CL/opencl.hpp>

#include "ocl_utils.h"
#include "ocl_image.h"
#include "ocl_svm_mat_allocator.h"

#define KERNEL_SPV      "kernel_4.spv"
#define KERNEL_PREFIX   "gpu_"

// **************************************************************************
// gpu_ function for kernel.
// Kernel name is automatically created from this function name
// removing prefix gpu_.
// 
// Kernel for BGR color rotation
// Kernel header from kernel*.cl:
// __kernel void convert_bgr_to_bw(          __global OCLImage *t_ocl_bgr_img,
//                                           __global OCLImage *t_ocl_bw_img )
cl_int gpu_convert_bgr_to_bw( cl::Program &t_program, OCLImage *t_ocl_bgr_img,
                                                      OCLImage *t_ocl_bw_img )
{
    cl_int l_err;

    // removing prefix gpu_
    std::string l_kern_name( __FUNCTION__ );
    if ( l_kern_name.find( KERNEL_PREFIX ) == 0 )
    {
        l_kern_name.erase( 0, strlen( KERNEL_PREFIX ) );
    }

    // select the kernel from opencl program
    cl::Kernel l_kern_convert_bgr_to_bw( t_program, l_kern_name.c_str(), &l_err );  CL_ERR_R( l_err );

    // set kernel arguments
    l_err = l_kern_convert_bgr_to_bw.setArg( 0, t_ocl_bgr_img );                CL_ERR_R( l_err );
    l_err = l_kern_convert_bgr_to_bw.setArg( 1, t_ocl_bw_img );                 CL_ERR_R( l_err );

    // list of SVM pointers for data synchronization
    l_kern_convert_bgr_to_bw.setSVMPointers( {
            t_ocl_bgr_img,
            t_ocl_bgr_img->m_data,
            t_ocl_bw_img,
            t_ocl_bw_img->m_data,
            } );

    // get default Queue
    cl::CommandQueue defQueue = cl::CommandQueue::getDefault();

    // size of workgroup, should be multiple of 64, so 256 is OK
    int l_wg_size_x = 16;
    int l_wg_size_y = 16;
    // global range
    int l_gr_size_x = ( t_ocl_bgr_img->m_size.x + ( l_wg_size_x - 1 ) ) / l_wg_size_x * l_wg_size_x;
    int l_gr_size_y = ( t_ocl_bgr_img->m_size.y + ( l_wg_size_y - 1 ) ) / l_wg_size_y * l_wg_size_y;

    // Submitting kernel for execution
    l_err = defQueue.enqueueNDRangeKernel( l_kern_convert_bgr_to_bw,
            // offset
            cl::NDRange( 0, 0 ),
            // global range
            cl::NDRange( l_gr_size_x, l_gr_size_y ),
            // work-group
            cl::NDRange( l_wg_size_x, l_wg_size_y ) );                          CL_ERR_R( l_err );

    // waiting for completion
    defQueue.finish();

    return CL_SUCCESS;
}

// **************************************************************************
int main( int t_narg, char **t_args )
{

    // check arguments
    if ( t_narg < 2 )
    {
        std::cerr << "Enter image name!" << std::endl;
        exit( EXIT_FAILURE );
    }

    cl_int l_err;

    l_err = ocl_init( 1 );                                                      CL_ERR_E( l_err );

    std::cout << "\nInitialization done." << std::endl;

    cl::Program l_program( ocl_load_program( KERNEL_SPV ) );

    if ( l_program() == nullptr )
    {
        std::cerr << "Program not built!" << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << "Program loaded.\n" << std::endl;

    // creating SVM allocator for cv::Mat
    SVMMatAllocator svmallocator;
    cv::Mat::setDefaultAllocator( &svmallocator );

    // load image from file
    cv::Mat l_cv_bgr_img = cv::imread( t_args[ 1 ], cv::IMREAD_UNCHANGED );

    // image read?
    if ( l_cv_bgr_img.empty() )
    {
        std::cerr << "Unable to open image '" << t_args[ 1 ] << "'." << std::endl;
        exit( EXIT_FAILURE );
    }

    // 3 or 4 channels?
    if ( l_cv_bgr_img.channels() != 4 )
    {
        // convert to 4 channels
        cv::cvtColor( l_cv_bgr_img, l_cv_bgr_img, cv::COLOR_BGR2BGRA );
    }

    // creating empty BW image, the same size as BGR image
    cv::Mat l_cv_bw_img( l_cv_bgr_img.size(), CV_8UC1 );

    // BGR OCLImage for kernel
    OCLImage *l_ocl_bgr_img = ocl_svm_malloc< OCLImage >();
    l_ocl_bgr_img->m_size.x = l_cv_bgr_img.size().width;
    l_ocl_bgr_img->m_size.y = l_cv_bgr_img.size().height;
    l_ocl_bgr_img->m_data = l_cv_bgr_img.data;

    // BW OCLImage for kernel
    OCLImage *l_ocl_bw_img = ocl_svm_malloc< OCLImage >();
    l_ocl_bw_img->m_size.x = l_cv_bw_img.size().width;
    l_ocl_bw_img->m_size.y = l_cv_bw_img.size().height;
    l_ocl_bw_img->m_data = l_cv_bw_img.data;

    // show loaded BGR image
    cv::imshow( "BGR Image", l_cv_bgr_img );

    // convert BGR image to BW image
    gpu_convert_bgr_to_bw( l_program, l_ocl_bgr_img, l_ocl_bw_img );

    // show new BW image
    cv::imshow( "BW Image", l_cv_bw_img );

    // wait for key
    cv::waitKey( 0 );
}

