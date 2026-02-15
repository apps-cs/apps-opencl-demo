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
 * @file ocl_svm_mat_allocator.cpp
 * @brief Share Virtual Memory Mat Allocator
 *
 * @details
 * Source file for cv::Mat Allocator class using Share Virtual Memory (SVM).
 * 
 ***************************************************************************/


#include "ocl_utils.h"
#include "ocl_svm_mat_allocator.h"

/// @copydoc SVMMatAllocator::allocate
cv::UMatData* SVMMatAllocator::allocate( 
        int dims, const int* sizes, int type,
        void* data0, size_t* step, cv::AccessFlag /*flags*/, cv::UMatUsageFlags /*usageFlags*/ ) const
{
    size_t total = CV_ELEM_SIZE( type );
    for( int i = dims-1; i >= 0; i-- )
    {
        if( step )
        {
            if( data0 && step[i] != CV_AUTOSTEP )
            {
                CV_Assert( total <= step[i] );
                total = step[i];
            }
            else
                step[i] = total;
        }
        total *= sizes[i];
    }
    uchar* data = data0 ? ( uchar* ) data0 : ocl_svm_malloc< uchar >( total );
    cv::UMatData* u = new cv::UMatData(this);
    u->data = u->origdata = data;
    u->size = total;
    if(data0)
        u->flags |= cv::UMatData::USER_ALLOCATED;
    return u;
}

/// @copydoc SVMMatAllocator::allocate
bool SVMMatAllocator::allocate( cv::UMatData* u, cv::AccessFlag /*accessFlags*/, cv::UMatUsageFlags /*usageFlags*/ ) const
{
    if( !u ) return false;
    return true;
}

/// @copydoc SVMMatAllocator::deallocate
void SVMMatAllocator::deallocate(cv::UMatData* u) const
{
    if( !u )
        return;

    CV_Assert( u->urefcount == 0 );
    CV_Assert( u->refcount == 0 );
    if( !( u->flags & cv::UMatData::USER_ALLOCATED ) )
    {
        ocl_svm_free( u->origdata );
        u->origdata = 0;
    }
    delete u;
}


