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

#include "ocl_image.h"

// kernel for creating chessboard
__kernel void create_chessboard( __global OCLImage *t_ocl_img, int t_sq_size )
{
    // get work-item position  
    size_t global_idx = get_global_id( 0 );
    size_t global_idy = get_global_id( 1 );

    // verify work-item position
    if ( global_idx >= t_ocl_img->m_size.x ) return;
    if ( global_idy >= t_ocl_img->m_size.y ) return;

    int l_sq_sx = t_sq_size * get_local_size( 0 );
    int l_sq_sy = t_sq_size * get_local_size( 1 );

    // odd or even index of chessboard square
    int l_sq_odd_even = global_idx / l_sq_sx + global_idy / l_sq_sy;

    // even square black, odd square white
    uchar l_bl_or_wh = 255 * ( l_sq_odd_even & 1 );

    // set point
    t_ocl_img->at4( global_idy, global_idx ) = { l_bl_or_wh, l_bl_or_wh, l_bl_or_wh, 0 };
}

// **************************************************************************
// kernel for creating dot image with alpha channel 
__kernel void create_transparent_dot( __global OCLImage *t_ocl_img, uchar4 t_color )
{
    // get work-item position  
    size_t global_idx = get_global_id( 0 );
    size_t global_idy = get_global_id( 1 );

    // verify work-item position
    if ( global_idx >= t_ocl_img->m_size.x ) return;
    if ( global_idy >= t_ocl_img->m_size.y ) return;

    // length of diagonal
    int l_diagonal = sqrt( ( float ) t_ocl_img->m_size.x * t_ocl_img->m_size.x +
                                     t_ocl_img->m_size.y * t_ocl_img->m_size.y );

    // relative positions of point from the center 
    int l_rx = global_idx - t_ocl_img->m_size.x / 2;
    int l_ry = global_idy - t_ocl_img->m_size.y / 2;

    // distance from the center
    int l_r = l_diagonal / 2 - sqrt( ( float ) l_rx * l_rx + l_ry * l_ry );

    // transparency decreases from the center
    t_color.w = 255 * l_r / ( l_diagonal / 2 );

    // set point
    t_ocl_img->at4( global_idy, global_idx ) = t_color;
}

// **************************************************************************
// kernel for inserting image into image
__kernel void insert_image( __global OCLImage *t_ocl_big_img, __global OCLImage *t_ocl_small_img, int2 t_position )
{
    // get work-item position, small image
    size_t global_idx = get_global_id( 0 );
    size_t global_idy = get_global_id( 1 );

    // verify work-item position, small image
    if ( global_idx >= t_ocl_small_img->m_size.x ) return;
    if ( global_idy >= t_ocl_small_img->m_size.y ) return;

    // position in big image
    int l_bx = t_position.x + global_idx;
    int l_by = t_position.y + global_idy;

    // position verification for big image
    if ( l_bx < 0 || l_bx >= t_ocl_big_img->m_size.x ) return;
    if ( l_by < 0 || l_by >= t_ocl_big_img->m_size.y ) return;

    // two corresponding points from big and small image
    uchar4 l_bg_bgr = t_ocl_big_img->at4( l_by, l_bx );
    uchar4 l_fg_bgr = t_ocl_small_img->at4( global_idy, global_idx );

    uchar4 l_out_bgr = { 0, 0, 0, 255 };
    // transparency calculation
    l_out_bgr.x = l_fg_bgr.x * l_fg_bgr.w / 255 + l_bg_bgr.x * ( 255 - l_fg_bgr.w ) / 255;
    l_out_bgr.y = l_fg_bgr.y * l_fg_bgr.w / 255 + l_bg_bgr.y * ( 255 - l_fg_bgr.w ) / 255;
    l_out_bgr.z = l_fg_bgr.z * l_fg_bgr.w / 255 + l_bg_bgr.z * ( 255 - l_fg_bgr.w ) / 255;

    // store result into big image
    t_ocl_big_img->at4( l_by, l_bx ) = l_out_bgr;
}



