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
 * @file ocl_utils.h
 * @brief OpenCL Utils for initialization, load program and SVM allocation.
 * 
 * @mainpage OpenCL Utils
 *
 * Main programming API:
 *
 * - @ref ocl_init -- @copybrief ocl_init
 *
 * - @ref ocl_load_program -- @copybrief ocl_load_program
 *
 * - @ref ocl_svm_malloc -- @copybrief ocl_svm_malloc
 *
 * - @ref ocl_svm_free -- @copybrief ocl_svm_free
 *
 * - @ref OCLImage -- @copybrief OCLImage
 * 
 * - @ref SVMMatAllocator -- @copybrief SVMMatAllocator
 *
 * 
 ***************************************************************************/

#ifndef __OCL_UTILS_H
#define __OCL_UTILS_H

#include <CL/opencl.hpp> 


/**
 * @name
 * @brief Macros for checking OpenCL Errors. 
 * @{
*/
#define CL_ERR_C( ERROR ) _CL_ERR( ERROR, ; )                                   //!< Display Error
#define CL_ERR_R( ERROR ) _CL_ERR( ERROR, return ( ERROR ); )                   //!< Display Error and return
#define CL_ERR_E( ERROR ) _CL_ERR( ERROR, exit( EXIT_FAILURE ); )               //!< Display Error and exit
/// @} 

// @cond 
#define _STREAM_ERROR( STREAM, ERROR, FUNCTION, LINE )               \
    _out_error( STREAM, ERROR, FUNCTION, LINE )

#define _PRINT_ERROR( ERROR, FUNCTION, LINE )                        \
    _STREAM_ERROR( std::cerr, ERROR, FUNCTION, LINE )

#define _CL_ERR( ERROR, CMD ) { if ( ( ERROR ) != CL_SUCCESS ) { _PRINT_ERROR( ERROR, __FUNCTION__, __LINE__ ); CMD } }

/* *
 * @brief Function is used internally to print error code
 * @param t_stream Output stream, usually cerr.
 * @param t_error Some cl_error. 
 * @param t_func_name Name of current function. 
 * @param t_line_num Line number in source code. 
*/
void _out_error( std::ostream &t_stream, int t_error, std::string t_func_name, int t_line_num );
// @endcond


/**
 * @anchor ocl_init
 * @brief OpenCL initialization.
 * 
 * @details
 * Function detect OpenCL environment. 
 * It detects how many Platforms are available and how many Devices 
 * are on the individual Platforms. 
 * The first Platform with GPU type Device is set as default. 
 *
 * When some GPU Device was found, then is created default Context and
 * default Queue. 
 *
 * After OpenCL initialization is available:
 * - cl::Platform::getDefault();
 * - cl::Device::getDefault();
 * - cl::Context::getDefault();
 * - cl::CommandQueue::getDefault();
 *
 * @param t_verbose Verbose mode of OpenCL initialization.
 * @param t_gpu_dev_index Index of selected GPU device, default 0
 * @return cl_int error code or CL_SUCCESS.
*/
cl_int ocl_init( int t_verbose = 0, int t_gpu_dev_index = 0 );


/**
 * @anchor ocl_load_program
 * @brief Function for loading program with kernels. 
 * @param t_kernel_filename File name with SPIRV code. 
 * @return Instance of cl::Program
*/
cl::Program ocl_load_program( const std::string t_kernel_filename );


/**
 * @anchor ocl_svm_malloc
 * @brief Function for easy SVM memory allocation. 
 * @param T data type.
 * @param t_size number of allocated elements.
 * @return pointer to allocated SVM memory. 
*/
template< typename T >
T* ocl_svm_malloc( size_t t_size = 1 ) 
{
    auto l_context = cl::Context::getDefault();
    if ( l_context() == nullptr ) 
    { 
        return nullptr; 
    }
    return (T*) clSVMAlloc( l_context(), CL_MEM_READ_WRITE, t_size, 0 );
}

/**
 * @anchor ocl_svm_free
 * @brief Function for SVM memory deallocation. 
 * @param t_ptr Pointer to SVM memory. 
*/
inline void ocl_svm_free( void *t_ptr ) 
{
    auto l_context = cl::Context::getDefault();
    if ( l_context() == nullptr ) 
    { 
        return; 
    }
    clSVMFree( l_context(), t_ptr );
}

#endif // __OCL_UTILS_H

