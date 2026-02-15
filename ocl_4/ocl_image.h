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
 * @file ocl_image.h
 * @brief This file contains structure \ref OCLImage for data transfer between 
 *   host and device. 
 *
 * @details
 * Header file for struct OCLImage. 
 * This structure is used for bidirectional transfer of data between 
 * host (PC) and device (GPU).
 * 
 ***************************************************************************/

#ifndef __OCL_IMAGE_H__
#define __OCL_IMAGE_H__


#ifndef __OPENCL_CPP_VERSION__
#include <CL/opencl.hpp>
#endif 

/**
 * @name
 * @brief Type unification for using in @ref OCLImage
 * @{
*/
#ifdef __OPENCL_CPP_VERSION__
    /// @name 
    /// @brief Types for OpenCL kernels
    /// @{
    using _uint4 = uint4;
    using _uchar4 = uchar4;
    using _uchar = uchar;
    /// @}
#else
    /// @name 
    /// @brief Types for CPP Source files
    /// @{
    using _uint4 = cl_uint4;
    using _uchar4 = cl_uchar4;
    using _uchar = cl_uchar;
    /// @}
#endif
/// @}


/**
 * @brief Structure for data transfer between host and device. 
*/
struct OCLImage
{
    _uint4 m_size;                  ///< Size of image: x - width, y - height
    
    /**
     * @brief Internal union allows to use more data types for one pointer.
    */
    union 
    {
        void *m_data;               ///< Anonymous pointer.
        _uchar4 *m_data4;           ///< Array of _uchar4 type.
        _uchar *m_data1;            ///< Array of _uchar type.
    };

    /**
     * Method returns refernece to one element of image using 2D coordinates.
     * @param t_y Vertical coordinates.
     * @param t_x Horizontal coordinates.
     * @return Reference to one element.
    */
    inline _uchar4 &at4( int t_y, int t_x ) 
    { 
        return m_data4[ m_size.x * t_y + t_x ]; 
    }

    /**
     * Method returns refernece to one element of image using 2D coordinates.
     * @param t_y Vertical coordinates.
     * @param t_x Horizontal coordinates.
     * @return Reference to one element.
    */
    inline _uchar &at1( int t_y, int t_x ) 
    { 
        return m_data1[ m_size.x * t_y + t_x ]; 
    }
};

#endif // __OCL_IMAGE_H__

