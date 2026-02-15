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
 * This demo program demonstrates OpenCL NDRange Execution Model.
 * Computation is organized in hierarchy:
 *
 * NDRange {Global Size}
 * +-- Work-Groups {Size}
 *     +-- Work-Items (kernels)
 * 
 ***************************************************************************/

#include <ostream>
#include <unistd.h>
#include <iostream>

#include <CL/opencl.hpp> 

#include "ocl_utils.h"

#define KERNEL_SPV      "kernel_1.spv"
#define KERNEL_PREFIX   "gpu_"

// **************************************************************************
// gpu_ function for kernel.
// Kernel name is automatically created from this function name
// removing prefix gpu_.
// NDRange execution model. 
cl_int gpu_ndrange_exec_model( cl::Program &t_program )
{
    cl_int l_err; 

    // removing prefix _gpu
    std::string l_kern_name( __FUNCTION__ );
    if ( l_kern_name.find( KERNEL_PREFIX ) == 0 )
    {
        l_kern_name.erase( 0, strlen( KERNEL_PREFIX ) );
    }

    // select the kernel from opencl program
    cl::Kernel l_kern_work_range( t_program, l_kern_name.c_str(), &l_err );     CL_ERR_R( l_err );

    // get default Queue
    cl::CommandQueue defQueue = cl::CommandQueue::getDefault();

    // Following Work-Group is not the correct size!
    // Recommended size of Work-Group should be multiple of 64!
    // NDRange is reduced to obtain short output. 
    // size of workgroup
    int l_wg_size_x = 3;
    int l_wg_size_y = 2;
    // global range 
    int l_gr_size_x = l_wg_size_x * 2;
    int l_gr_size_y = l_wg_size_y * 4;
    
    // Submitting kernel for execution
    l_err = defQueue.enqueueNDRangeKernel( l_kern_work_range, 
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
int main()
{
    cl_int l_err;

    l_err = ocl_init( 2, 0 );                                                   CL_ERR_E( l_err );

    std::cout << "\nInitialization done." << std::endl;

    cl::Program l_program( ocl_load_program( KERNEL_SPV ) );

    if ( l_program() == nullptr )
    {
        std::cerr << "Program not built!" << std::endl;
        exit( EXIT_FAILURE );
    }

    std::cout << "Program loaded.\n" << std::endl;

    std::cout << "NDRange Execution Model:" << std::endl;

    gpu_ndrange_exec_model( l_program );
}


