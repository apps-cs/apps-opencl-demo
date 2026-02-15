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
 * Animation with bouncing ball.
 *
 ***************************************************************************/

#include <cstdlib>
#include <ostream>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>

#include <CL/opencl.hpp>

#include "ocl_utils.h"
#include "ocl_image.h"
#include "ocl_svm_mat_allocator.h"

#define KERNEL_SPV      "kernel_6.spv"
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

int main( int t_narg, char **t_args )
{
    if ( t_narg < 2 )
    {
        std::cout << "Enter image name!" << std::endl;
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

    // creating empty image
    cv::Mat l_cv_background_img( IMG_SIZEY, IMG_SIZEX, CV_8UC4 );

    // Background OCLImage for kernel
    OCLImage *l_ocl_background_img = ocl_svm_malloc< OCLImage >();
    l_ocl_background_img->m_size.x = l_cv_background_img.size().width;
    l_ocl_background_img->m_size.y = l_cv_background_img.size().height;
    l_ocl_background_img->m_data = l_cv_background_img.data;

    gpu_create_chessboard( l_program, l_ocl_background_img, 3 );
    
    // show created chessboard
    cv::imshow( "Chessboard", l_cv_background_img );

    cv::Mat l_cv_load_img;

    std::cout << "Opening image: '" << t_args[ 1 ] << "'." << std::endl;

    l_cv_load_img = cv::imread( t_args[ 1 ], cv::IMREAD_UNCHANGED );

    if ( !l_cv_load_img.empty() && l_cv_load_img.channels() == 4 )
    {
        std::cout << "Image loaded." << std::endl;
    }
    else if ( l_cv_load_img.channels() != 4 )
    {
        std::cout << "Image is not transparent!" << std::endl;        
        exit( EXIT_FAILURE );
    }
    else
    {
        std::cout << "Unable to read image!" << std::endl; 
        exit( EXIT_FAILURE );
    }

    OCLImage *l_ocl_load_img = ocl_svm_malloc< OCLImage >();
    l_ocl_load_img->m_size.x = l_cv_load_img.size().width;
    l_ocl_load_img->m_size.y = l_cv_load_img.size().height;
    l_ocl_load_img->m_data = l_cv_load_img.data;

    // copy of background
    cv::Mat l_cv_bg_backup_img;
    l_cv_background_img.copyTo( l_cv_bg_backup_img );

    // animation
    // positive z axis is up
    float anim_g = -9.81; 
    // one m is 1000 pixel
    int anim_ppm = 1000;
    // starting position is above background image
    float anim_sz0 = ( float ) l_cv_background_img.rows / anim_ppm;
    // formula: s = 0.5 * a * t * t
    // time of the first cycle
    float anim_tc = sqrtf( anim_sz0 / ( 0.5 * - anim_g ) );
    // starting speed
    float anim_vz0 = 0;

    timeval anim_tv_start;
    gettimeofday( &anim_tv_start, nullptr );

    while ( 1 )
    {
        timeval anim_tv_cur, anim_tv_delta;
        gettimeofday( &anim_tv_cur, nullptr );
        timersub( &anim_tv_cur, &anim_tv_start, &anim_tv_delta );

        // current time in cycle
        float anim_t = anim_tv_delta.tv_sec + anim_tv_delta.tv_usec / 1000000.0;

        // new z position (height)
        float anim_z = anim_sz0 + anim_vz0 * anim_t + 0.5 * anim_g * anim_t * anim_t;

        cl_int2 ipos = {{ IMG_SIZEX / 2, 0 }};
        ipos.y = l_ocl_background_img->m_size.y - anim_z * anim_ppm - l_ocl_load_img->m_size.y;

        // restore background
        l_cv_bg_backup_img.copyTo( l_cv_background_img );

        gpu_insert_image( l_program, l_ocl_background_img, l_ocl_load_img, ipos );

        cv::imshow( "Chessboard", l_cv_background_img );
        cv::waitKey( 1 );
    
        // one cycle passed
        if ( anim_t > anim_tc )
        {
            if ( anim_vz0 == 0 ) anim_vz0 = - anim_g * anim_tc; // speed upon impact
            anim_vz0 = anim_vz0 * 0.8;          // impact reduces energy/speed
            anim_t = anim_t - anim_tc;          // time to next cycle
            anim_tc = - 2 * anim_vz0 / anim_g;  // new time cycle (up/down)
            anim_sz0 = 0;                       // starting position is at the bottom.
            if ( anim_tc < 0.01 ) break;        // good time to stop animation

            // new cycle starting tv_time
            anim_tv_delta.tv_sec = ( int ) anim_t;
            anim_tv_delta.tv_usec = ( anim_t - ( int ) anim_t ) * 1000000;
            timeradd( &anim_tv_cur, &anim_tv_delta, &anim_tv_start );
        }
    }
   
    // wait for key
    cv::waitKey( 0 );
}

