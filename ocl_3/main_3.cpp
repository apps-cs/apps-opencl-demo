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
 * Rotation of BGR->RGB colors in image. 
 * Program will create image with color gradient or load image from a file. 
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

#define KERNEL_SPV      "kernel_3.spv"
#define KERNEL_PREFIX   "gpu_"

// **************************************************************************
// gpu_ function for kernel.
// Kernel name is automatically created from this function name
// removing prefix gpu_.
// 
// BGR colors rotation.
// Kernel header from kernel*.cl:
//__kernel void rotate_bgr(            __global OCLImage *t_ocl_img )
cl_int gpu_rotate_bgr( cl::Program &t_program,  OCLImage *t_ocl_img )
{
    cl_int l_err;

    // removing prefix gpu_
    std::string l_kern_name( __FUNCTION__ );
    if ( l_kern_name.find( KERNEL_PREFIX ) == 0 )
    {
        l_kern_name.erase( 0, strlen( KERNEL_PREFIX ) );
    }

    // select the kernel from opencl program
    cl::Kernel l_kern_rotate_bgr( t_program, l_kern_name.c_str(), &l_err );      CL_ERR_R( l_err );

    // set kernel arguments
    l_err = l_kern_rotate_bgr.setArg( 0, t_ocl_img );                            CL_ERR_R( l_err );

    // list of SVM pointers for data synchronization
    l_kern_rotate_bgr.setSVMPointers( { t_ocl_img, t_ocl_img->m_data } );

    // get default Queue
    cl::CommandQueue defQueue = cl::CommandQueue::getDefault();

    // size of workgroup, should be multiple of 64, so 256 is OK 
    int l_wg_size_x = 16;
    int l_wg_size_y = 16;
    // global range 
    int l_gr_size_x = ( t_ocl_img->m_size.x + ( l_wg_size_x - 1 ) ) / l_wg_size_x * l_wg_size_x;
    int l_gr_size_y = ( t_ocl_img->m_size.y + ( l_wg_size_y - 1 ) ) / l_wg_size_y * l_wg_size_y;
    
    // Submitting kernel for execution
    l_err = defQueue.enqueueNDRangeKernel( l_kern_rotate_bgr, 
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
#define IMG_SIZEX   432
#define IMG_SIZEY   321

int main( int t_narg, char **t_args )
{
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

    // creating empty image
    cv::Mat l_cv_img( IMG_SIZEY, IMG_SIZEX, CV_8UC4 );

    // load image from file?
    if ( t_narg > 1 )
    {
        // load image 
        l_cv_img = cv::imread( t_args[ 1 ], cv::IMREAD_UNCHANGED );

        // image read?
        if ( l_cv_img.empty() )
        {
            std::cerr << "Unable to open image '" << t_args[ 1 ] << "'." << std::endl;
            exit( EXIT_FAILURE );
        }

        // 3 or 4 channels?
        if ( l_cv_img.channels() != 4 )
        {
            // convert to 4 channels
            cv::cvtColor( l_cv_img, l_cv_img, cv::COLOR_BGR2BGRA );
        }
    }
    else
    {   // creating image as RGBx gradient
        for ( int y = 0; y < l_cv_img.rows; y++ )
        {
            for ( int x  = 0; x < l_cv_img.cols; x++ )
            {
                int l_dx = x - l_cv_img.cols / 2;

                // gradient
                unsigned char l_grad = 255 * abs( l_dx ) / ( l_cv_img.cols / 2 );
                unsigned char l_inv_grad = 255 - l_grad;

                // left or right half of the gradient
                cl_uchar4 l_bgr = ( l_dx < 0 ) ? ( cl_uchar4 ) {{ l_grad, l_inv_grad, 0, 0 }} : ( cl_uchar4 ) {{ 0, l_inv_grad, l_grad, 0 }};

                // put pixel into image
                cv::Vec4b l_v4bgr( l_bgr.x, l_bgr.y, l_bgr.z );
                l_cv_img.at<cv::Vec4b>( y, x ) = l_v4bgr;
                // also possible: cv_img.at<cl_uchar4>( y, x ) = bgr;
            }
        }
    }

    // OCLImage for kernel
    OCLImage *l_ocl_img = ocl_svm_malloc< OCLImage >();
    l_ocl_img->m_size.x = l_cv_img.size().width;
    l_ocl_img->m_size.y = l_cv_img.size().height;
    l_ocl_img->m_data = l_cv_img.data;

    // show loaded/created image
    cv::imshow( "B-G-R Image", l_cv_img );

    // rotate color
    gpu_rotate_bgr( l_program, l_ocl_img );

    // show new image
    cv::imshow( "B-G-R Image & Color Rotation", l_cv_img );

    // wait for key
    cv::waitKey( 0 );
}

