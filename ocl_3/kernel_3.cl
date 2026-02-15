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

#include "ocl_image.h"

// kernel for BGR color rotation
__kernel void rotate_bgr( __global OCLImage *t_ocl_img )
{
    // get work-item position  
    size_t global_idx = get_global_id( 0 );
    size_t global_idy = get_global_id( 1 );

    // verify work-item position
    if ( global_idx >= t_ocl_img->m_size.x ) return;
    if ( global_idy >= t_ocl_img->m_size.y ) return;

    // get one point from image
    uchar4 l_bgr = t_ocl_img->at4( global_idy, global_idx );

    // rotate colors
    uchar4 l_bgr_rot;
    l_bgr_rot.x = l_bgr.y;
    l_bgr_rot.y = l_bgr.z;
    l_bgr_rot.z = l_bgr.x;

    // put point into image
    t_ocl_img->at4( global_idy, global_idx ) = l_bgr_rot;
}


