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
 * This demo program demostrates OpenCL NDRange Execution Model.
 * Computation is organized in hierarchy:
 *
 * NDRange {Global Size}
 * +-- Work-Groups {Size}
 *     +-- Work-Items (kernels)
 * 
 ***************************************************************************/

// kernel demostrates NDRange Execution Model
__kernel void ndrange_exec_model()
{
    // NDRange
    size_t global_sx = get_global_size( 0 );
    size_t global_sy = get_global_size( 1 );
    size_t global_idx = get_global_id( 0 );
    size_t global_idy = get_global_id( 1 );

    // Work-Group 
    size_t group_sx = get_num_groups( 0 );
    size_t group_sy = get_num_groups( 1 );
    size_t group_idx = get_group_id( 0 );
    size_t group_idy = get_group_id( 1 );

    // Local Items of Work-Group
    size_t local_sx = get_local_size( 0 );
    size_t local_sy = get_local_size( 1 );
    size_t local_idx = get_local_id( 0 );
    size_t local_idy = get_local_id( 1 );

    printf( "Global Range {%2zd,%2zd}[%2zd,%2zd] Work-group {%2zd,%2zd}[%2zd,%2zd] Local {%2zd,%2zd}[%2zd,%2zd] \n", 
            global_sx, global_sy, 
            global_idx, global_idy, 
            group_sx, group_sy,
            group_idx, group_idy,
            local_sx, local_sy,
            local_idx, local_idy
            );
}


