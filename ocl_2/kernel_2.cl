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

// kernel for parallel multiplication by scalar
__kernel void mult_vect( __global float *t_vector, float t_mult, int t_len )
{
    // get work-item positon in global range
    size_t global_idx = get_global_id( 0 );

    // verify work-item position
    if ( global_idx >= t_len ) return;

    // multiplication of one element
    t_vector[ global_idx ] *= t_mult;
}


