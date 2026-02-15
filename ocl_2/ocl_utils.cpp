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
 * @file ocl_utils.cpp
 * @brief OpenCL Utils for initialization, load program and SVM allocation.
 * 
 ***************************************************************************/

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <filesystem>

#include <CL/opencl.hpp> 

#include "ocl_utils.h"

/// @copydoc _out_error
void _out_error( std::ostream &t_stream, int t_error, std::string t_func_name, int t_line_num )
{
    t_stream << 
        "Error: " << t_error << 
        " in function '" << t_func_name << 
        "' on line "<< t_line_num << "." << std::endl;
}


// @copydoc ocl_init
cl_int ocl_init( int t_verbose, int t_gpu_dev_index )
{
    const char * l_dev_types[ 17 ] = 
        { nullptr, "DEFAULT", "CPU", nullptr, "GPU", nullptr, nullptr, nullptr, "ACCELERATOR", 
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "CUSTOM" };

    cl_int l_err;

    // Searching of platforms
    std::vector<cl::Platform> l_platforms;
    l_err = cl::Platform::get( &l_platforms );                                  CL_ERR_R( l_err );

    // No platforms
    if ( l_platforms.size() == 0 )
    {
        std::cerr << "No OpenCL 3.x platform found!" << std::endl;
        exit( EXIT_FAILURE );
    }

    std::vector< std::pair< cl::Platform, cl::Device > > l_gpu_devices;

    // variables for formating verbose output
    int l_left = 40;
    int l_shift = 0;
    int l_indent = 4;

    if ( t_verbose > 1  )
    {
        std::cout << std::setw(l_left) << std::left << "Platforms " << l_platforms.size() << std::endl;
    }

    for ( auto ipla = 0; ipla < l_platforms.size(); ipla++ )
    {
        cl::Platform &p = l_platforms[ ipla ];

        // Search of devices
        std::vector<cl::Device> l_devices;
        p.getDevices( CL_DEVICE_TYPE_ALL, &l_devices );

        for ( auto &d : l_devices )
        {
            if ( d.getInfo< CL_DEVICE_TYPE >() == CL_DEVICE_TYPE_GPU && 
                    p.getInfo< CL_PLATFORM_VERSION >().find( "OpenCL 3." ) >= 0 )
            {
                l_gpu_devices.push_back( { p, d } ); 
            }
        }
        

        // print information about platforms and devices
        if ( t_verbose > 1 )
        { // print
            l_shift += l_indent;
            l_left -= l_indent;

            std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Platform" << "[" << ipla << "]" << std::endl;

            std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Platform Name"     << p.getInfo< CL_PLATFORM_NAME >() << std::endl;
            std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Platform Vendor"   << p.getInfo< CL_PLATFORM_VENDOR >() << std::endl;
            std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Platform Version"  << p.getInfo< CL_PLATFORM_VERSION >() << std::endl;

            std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Devices" << l_devices.size() << std::endl;

            for ( auto idev = 0; idev < l_devices.size(); idev++ )
            {
                cl::Device &d = l_devices[ idev ];

                l_shift += l_indent;
                l_left -= l_indent;

                std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Device" << "[" << idev << "]" << std::endl;

                std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Device Name"     << d.getInfo< CL_DEVICE_NAME >() << std::endl;
                std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Device Vendor"   << d.getInfo< CL_DEVICE_VENDOR >() << std::endl;
                std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Device Version"  << d.getInfo< CL_DEVICE_VERSION >() << std::endl;
                std::cout << std::setw( l_shift ) << "" << std::setw( l_left ) << std::left << "Device Type"     << l_dev_types[ d.getInfo< CL_DEVICE_TYPE >() ] << std::endl;

                l_shift -= l_indent;
                l_left += l_indent;
            }

            l_shift -= l_indent;
            l_left += l_indent;
        } // end print
    }

    // An OpenCL available?
    if ( l_gpu_devices.size() == 0 )
    {
        std::cerr << "No OpenCL 3.x device found!" << std::endl;
        exit( EXIT_FAILURE );
    }

    if ( l_gpu_devices.size() <= t_gpu_dev_index )
    {
        std::cerr << "Only " << l_gpu_devices.size() << " GPU Devices detected. ";
        std::cerr << "Device [" << t_gpu_dev_index << "] can't be selected!" << std::endl;
        exit( EXIT_FAILURE );
    }

    if ( t_verbose > 0 )
    {
        std::cout << "Found " << l_gpu_devices.size() << " GPU Devices." << std::endl;
        std::cout << "Device [" <<  t_gpu_dev_index << "] will be used." << std::endl;
    }

    auto l_pair = l_gpu_devices[ t_gpu_dev_index ];

    // set global default platform and device
    cl::Platform::setDefault( l_pair.first );
    cl::Device::setDefault( l_pair.second );

    if ( t_verbose > 0 )
    {
        std::cout << "Default Platform created." << std::endl;
        std::cout << "Default Device created." << std::endl;
    }

    cl_device_svm_capabilities caps = l_pair.second.getInfo< CL_DEVICE_SVM_CAPABILITIES > ();
    if ( ( caps &  CL_DEVICE_SVM_COARSE_GRAIN_BUFFER ) == 0 )
    {
        std::cerr << "Share Virtual Memory (SVM) not supported!" << std::endl;
        exit( EXIT_FAILURE );
    }
    
    // create default context
    cl_context_properties l_prop[] = { CL_CONTEXT_PLATFORM, ( cl_context_properties ) l_pair.first(), 0 };
    cl::Context defCont( l_pair.second, l_prop, nullptr, nullptr, &l_err );     CL_ERR_R( l_err );
    cl::Context::setDefault( defCont );

    if ( t_verbose > 0 )
    {
        std::cout << "Default Context created." << std::endl;
    }

    cl::CommandQueue defQueue( ( cl_command_queue_properties ) 0U, &l_err );    CL_ERR_R( l_err );
    cl::CommandQueue::setDefault( defQueue );

    if ( t_verbose > 0 )
    {
        std::cout << "Default Queue created." << std::endl;
    }

    return CL_SUCCESS;
}


// @copydoc ocl_load_program
cl::Program ocl_load_program( const std::string t_kernel_filename )
{
    cl::Program l_program;

    // get size of SPIRV file 
    decltype( std::filesystem::file_size( "" ) ) l_filesize;
    try 
    {
        l_filesize = std::filesystem::file_size( t_kernel_filename );
    }
    catch ( std::filesystem::filesystem_error& e)
    {
        std::cerr << "Filesize '" << t_kernel_filename << "' error: " << e.what() << std::endl;
        return l_program;
    }

    // allocate space for file and read SPIRV code
    std::vector< char > l_spirv_data( l_filesize );
    std::ifstream l_spirv_istr( t_kernel_filename );
    l_spirv_istr.read( l_spirv_data.data(), l_filesize );
    if ( l_spirv_istr.gcount() != l_filesize )
    {
        std::cerr << "Unable to read file `" << t_kernel_filename << "." << std::endl;
        l_spirv_istr.close();
        return l_program;
    }
    l_spirv_istr.close();
    // program loaded
    
    // build program with kernels
    cl_int l_err;
    l_program = cl::Program( cl::Context::getDefault(), l_spirv_data, true, &l_err ); CL_ERR_C( l_err );

    if ( l_err != CL_SUCCESS )
    {
        std::cerr << "Build of '" << t_kernel_filename << "' failed!" << std::endl;
        auto out = l_program.getBuildInfo< CL_PROGRAM_BUILD_LOG >( &l_err );
        for (auto &pair : out) 
        {
            std::cerr << pair.second << std::endl << std::endl;
        }
        return l_program;
    }
    // build sucessfull
    
    return l_program;
}


