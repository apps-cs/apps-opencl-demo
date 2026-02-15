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
 * Creating (and loading) transparent image. 
 * Inserting transparent image into image. 
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

#define KERNEL_SPV      "kernel_5.spv"
#define KERNEL_PREFIX   "gpu_"

// **************************************************************************
// gpu_ function for kernel.
// Kernel name is automatically created from this function name
// removing prefix gpu_.
//
// Kernel for creating chessboard
// Kernel header from kernel*.cl:
// __kernel void create_chessboard(          __global OCLImage *t_ocl_img, 
//                                                    int t_sq_size )
cl_int gpu_create_chessboard( cl::Program &t_program, OCLImage *t_ocl_img,
                                                      int t_sq_size )
{
    cl_int l_err;

    // removing prefix gpu_
    std::string l_kern_name( __FUNCTION__ );
    if ( l_kern_name.find( KERNEL_PREFIX ) == 0 )
    {
        l_kern_name.erase( 0, strlen( KERNEL_PREFIX ) );
    }

    // select the kernel from opencl program
    cl::Kernel l_kern_create_chessboard( t_program, l_kern_name.c_str(), &l_err );  CL_ERR_R( l_err );

    // set kernel arguments
    l_err = l_kern_create_chessboard.setArg( 0, t_ocl_img );                    CL_ERR_R( l_err );
    l_err = l_kern_create_chessboard.setArg( 1, t_sq_size );                    CL_ERR_R( l_err );

    // list of SVM pointers for data synchronization
    l_kern_create_chessboard.setSVMPointers( {
            t_ocl_img,
            t_ocl_img->m_data,
            } );

    // get default Queue
    cl::CommandQueue defQueue = cl::CommandQueue::getDefault();

    // size of workgroup, should be multiple of 64, so 256 is OK
    int l_wg_size_x = 16;
    int l_wg_size_y = 16;
    // global range
    int l_gr_size_x = ( t_ocl_img->m_size.x + ( l_wg_size_x - 1 ) ) / l_wg_size_x * l_wg_size_x;
    int l_gr_size_y = ( t_ocl_img->m_size.y + ( l_wg_size_y - 1 ) ) / l_wg_size_y * l_wg_size_y;

    // Submitting kernel for execution
    l_err = defQueue.enqueueNDRangeKernel( l_kern_create_chessboard,
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
// gpu_ function for kernel.
// Kernel name is automatically created from this function name
// removing prefix gpu_.
//
// Kernel for creating dot image with alpha channel.
// Kernel header from kernel*.cl:
// __kernel void create_transparent_dot(          __global OCLImage *t_ocl_img, 
//                                                         uchar4 t_color )
cl_int gpu_create_transparent_dot( cl::Program &t_program, OCLImage *t_ocl_img,
                                                           cl_uchar4 t_color )
{
    cl_int l_err;

    // removing prefix gpu_
    std::string l_kern_name( __FUNCTION__ );
    if ( l_kern_name.find( KERNEL_PREFIX ) == 0 )
    {
        l_kern_name.erase( 0, strlen( KERNEL_PREFIX ) );
    }

    // select the kernel from opencl program
    cl::Kernel l_kern_transparent_dot( t_program, l_kern_name.c_str(), &l_err );  CL_ERR_R( l_err );

    // set kernel arguments
    l_err = l_kern_transparent_dot.setArg( 0, t_ocl_img );                      CL_ERR_R( l_err );
    l_err = l_kern_transparent_dot.setArg( 1, t_color );                        CL_ERR_R( l_err );

    // list of SVM pointers for data synchronization
    l_kern_transparent_dot.setSVMPointers( {
            t_ocl_img,
            t_ocl_img->m_data,
            } );

    // get default Queue
    cl::CommandQueue defQueue = cl::CommandQueue::getDefault();

    // size of workgroup, should be multiple of 64, so 256 is OK
    int l_wg_size_x = 16;
    int l_wg_size_y = 16;
    // global range
    int l_gr_size_x = ( t_ocl_img->m_size.x + ( l_wg_size_x - 1 ) ) / l_wg_size_x * l_wg_size_x;
    int l_gr_size_y = ( t_ocl_img->m_size.y + ( l_wg_size_y - 1 ) ) / l_wg_size_y * l_wg_size_y;

    // Submitting kernel for execution
    l_err = defQueue.enqueueNDRangeKernel( l_kern_transparent_dot,
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
// gpu_ function for kernel.
// Kernel name is automatically created from this function name
// removing prefix gpu_.
//
// Kernel for inserting image into image
// Kernel header from kernel*.cl:
// __kernel void insert_image(          __global OCLImage *t_ocl_big_img, 
//                                      __global OCLImage *t_ocl_small_img, 
//                                               int2 t_position )
cl_int gpu_insert_image( cl::Program &t_program, OCLImage *t_ocl_big_img,
                                                 OCLImage *t_ocl_small_img,
                                                 cl_int2 t_position )
{
    cl_int l_err;

    // removing prefix gpu_
    std::string l_kern_name( __FUNCTION__ );
    if ( l_kern_name.find( KERNEL_PREFIX ) == 0 )
    {
        l_kern_name.erase( 0, strlen( KERNEL_PREFIX ) );
    }

    // select the kernel from opencl program
    cl::Kernel l_kern_insert_image( t_program, l_kern_name.c_str(), &l_err );  CL_ERR_R( l_err );

    // set kernel arguments
    l_err = l_kern_insert_image.setArg( 0, t_ocl_big_img );                     CL_ERR_R( l_err );
    l_err = l_kern_insert_image.setArg( 1, t_ocl_small_img );                   CL_ERR_R( l_err );
    l_err = l_kern_insert_image.setArg( 2, t_position );                        CL_ERR_R( l_err );

    // list of SVM pointers for data synchronization
    l_kern_insert_image.setSVMPointers( {
            t_ocl_big_img,
            t_ocl_big_img->m_data,
            t_ocl_small_img,
            t_ocl_small_img->m_data,
            } );

    // get default Queue
    cl::CommandQueue defQueue = cl::CommandQueue::getDefault();

    // size of workgroup, should be multiple of 64, so 256 is OK
    int l_wg_size_x = 16;
    int l_wg_size_y = 16;
    // global range
    int l_gr_size_x = ( t_ocl_small_img->m_size.x + ( l_wg_size_x - 1 ) ) / l_wg_size_x * l_wg_size_x;
    int l_gr_size_y = ( t_ocl_small_img->m_size.y + ( l_wg_size_y - 1 ) ) / l_wg_size_y * l_wg_size_y;

    // Submitting kernel for execution
    l_err = defQueue.enqueueNDRangeKernel( l_kern_insert_image,
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
#define IMG_SIZEX   876
#define IMG_SIZEY   765

#define DOT_SIZE    300

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
    cv::Mat l_cv_background_img( IMG_SIZEY, IMG_SIZEX, CV_8UC4 );

    // Background OCLImage for kernel
    OCLImage *l_ocl_background_img = ocl_svm_malloc< OCLImage >();
    l_ocl_background_img->m_size.x = l_cv_background_img.size().width;
    l_ocl_background_img->m_size.y = l_cv_background_img.size().height;
    l_ocl_background_img->m_data = l_cv_background_img.data;

    gpu_create_chessboard( l_program, l_ocl_background_img, 3 );
    
    // show created chessboard
    cv::imshow( "I. Chessboard", l_cv_background_img );

    // creating cv::Mat for transparent dot
    cv::Mat l_ocl_transp_dot( DOT_SIZE, DOT_SIZE, CV_8UC4 );
    OCLImage *l_ocl_dot_img = ocl_svm_malloc< OCLImage >();
    l_ocl_dot_img->m_size.x = l_ocl_transp_dot.size().width;
    l_ocl_dot_img->m_size.y = l_ocl_transp_dot.size().height;
    l_ocl_dot_img->m_data = l_ocl_transp_dot.data;

    // generating chessboard image
    gpu_create_transparent_dot( l_program, l_ocl_dot_img, {{ 0, 0, 255, 0 }} );

    // inserting transparent image into chessboard image
    gpu_insert_image( l_program, l_ocl_background_img, l_ocl_dot_img, {{ 100, 50 }} );

    // show dot 
    cv::imshow( "II. Dot", l_ocl_transp_dot );
    // show chessboard with dot
    cv::imshow( "III. Chessboard with Dot", l_cv_background_img );

    // was image file entered?
    if ( t_narg > 1 )
    {
        std::cout << "Opening image: '" << t_args[ 1 ] << "'." << std::endl;

        cv::Mat l_cv_load_img = cv::imread( t_args[ 1 ], cv::IMREAD_UNCHANGED );

        if ( !l_cv_load_img.empty() && l_cv_load_img.channels() == 4 )
        {
            std::cout << "Image loaded." << std::endl;

            OCLImage *l_ocl_load_img = ocl_svm_malloc< OCLImage >();
            l_ocl_load_img->m_size.x = l_cv_load_img.size().width;
            l_ocl_load_img->m_size.y = l_cv_load_img.size().height;
            l_ocl_load_img->m_data = l_cv_load_img.data;

            // insert new transparent image into chessboard image
            gpu_insert_image( l_program, l_ocl_background_img, l_ocl_load_img, {{ IMG_SIZEX / 2, IMG_SIZEY / 2 }} );

            cv::imshow( "IV. Chessboard with loaded transparent image", l_cv_background_img );
        }
        else if ( l_cv_load_img.channels() != 4 )
        {
            std::cout << "Image is not transparent!" << std::endl;        
        }
        else
        {
            std::cout << "Unable to read image!" << std::endl; 
        }
    }

    // wait for key
    cv::waitKey( 0 );
}

