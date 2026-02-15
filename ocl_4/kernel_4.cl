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

#include "ocl_image.h"

// kernel for BGR color rotation
__kernel void convert_bgr_to_bw( __global OCLImage *t_ocl_bgr_img, __global OCLImage *t_ocl_bw_img )
{
    // get work-item position  
    size_t global_idx = get_global_id( 0 );
    size_t global_idy = get_global_id( 1 );

    // verify work-item position
    if ( global_idx >= t_ocl_bgr_img->m_size.x ) return;
    if ( global_idy >= t_ocl_bgr_img->m_size.y ) return;

    // get one point from image
    uchar4 l_bgr = t_ocl_bgr_img->at4( global_idy, global_idx );

    // convert BGR to BW: 10% Blue + 59% Green + 30% Red
    //uchar l_bw = l_bgr.x * 0.11f + l_bgr.y * 0.59f + l_bgr.z * 0.30f;
    uchar l_bw = l_bgr.x * 11 / 100 + l_bgr.y * 59 / 100 + l_bgr.z * 30 / 100;

    // put point into image
    t_ocl_bw_img->at1( global_idy, global_idx ) = l_bw;
}


