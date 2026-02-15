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
 * Parallel multiplication of a vector by scalar.
 * 
 ***************************************************************************/

#include <cstdlib>
#include <ostream>
#include <unistd.h>
#include <iostream>
#include <math.h>

#include <CL/opencl.hpp> 

#include "ocl_utils.h"

#define KERNEL_SPV      "kernel_2.spv"
#define KERNEL_PREFIX   "gpu_"

// **************************************************************************
// gpu_ function for kernel.
// Kernel name is automatically created from this function name
// removing prefix gpu_.
// 
// Parallel multiplication of vector by scalar. 
// Kernel header:
//__kernel void mult_vect(           __global float *t_vector, float t_mult, int t_len )
cl_int gpu_mult_vect( cl::Program &t_program, float *t_vector, float t_mult, int t_len )
{
    cl_int l_err;

    // removing prefix gpu_
    std::string l_kern_name( __FUNCTION__ );
    if ( l_kern_name.find( KERNEL_PREFIX ) == 0 )
    {
        l_kern_name.erase( 0, strlen( KERNEL_PREFIX ) );
    }

    // select the kernel from opencl program
    cl::Kernel l_kern_mult_vect( t_program, l_kern_name.c_str(), &l_err );      CL_ERR_R( l_err );

    // set kernel arguments
    l_err = l_kern_mult_vect.setArg( 0, t_vector );                             CL_ERR_R( l_err );
    l_err = l_kern_mult_vect.setArg( 1, t_mult );                               CL_ERR_R( l_err );
    l_err = l_kern_mult_vect.setArg( 2, t_len );                                CL_ERR_R( l_err );

    // list of SVM pointers for data synchronization
    l_kern_mult_vect.setSVMPointers( { t_vector } );

    // get default Queue
    cl::CommandQueue defQueue = cl::CommandQueue::getDefault();

    // size of workgroup, should be multiple of 64
    int l_wg_size_x = 128;
    // global range 
    int l_gr_size_x = ( t_len + ( l_wg_size_x - 1 ) ) / l_wg_size_x * l_wg_size_x;
    
    // Submitting kernel for execution
    l_err = defQueue.enqueueNDRangeKernel( l_kern_mult_vect, 
            // offset
            cl::NDRange( 0, 0 ), 
            // global range
            cl::NDRange( l_gr_size_x ), 
            // work-group
            cl::NDRange( l_wg_size_x ) );                                       CL_ERR_R( l_err );
    
    // waiting for completion
    defQueue.finish();

    return CL_SUCCESS;
}

// **************************************************************************
int main()
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

    // length of vector
    int N = 333;

    // vector allocation in SVM memory
    float *l_vector = ocl_svm_malloc<float>( N );

    if ( l_vector == nullptr ) 
    {
        std::cerr << "Vector not allocated!" << std::endl;
        exit( EXIT_FAILURE );
    }

    // vector initialization
    for ( int i = 0; i < N; i++ )
    {
        l_vector[ i ] = i;
    }
    
    std::cout << "Vector allocated and initialized." << std::endl;

    gpu_mult_vect( l_program, l_vector, M_PI, N );                              CL_ERR_E( l_err );

    std::cout << "Result of vector multiplication:" << std::endl;

    for ( int i = 0; i < N; i++ )
    {
        std::cout << "[" << i << "] = " << l_vector[ i ] << std::endl;
    }
}

