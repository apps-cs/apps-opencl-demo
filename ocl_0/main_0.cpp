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
 * This demo program is used the only for verification of installed 
 * compilers and OpenCL files. 
 * 
 ***************************************************************************/

#include <ostream>
#include <unistd.h>
#include <iostream>

#include <CL/opencl.hpp> 

#include "ocl_utils.h"

int main()
{
    cl_int l_err;

    l_err = ocl_init( 2, 0 );                                                   CL_ERR_E( l_err );

    std::cout << "\nOpenCL 3.0 available, at least one Platform and one GPU Device detected." << std::endl;
}
