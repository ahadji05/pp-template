#include <iostream>
#include "ppt/fourier/FourierCuda.hpp"
#include "ppt/routines/utils.hpp"
#include "ppt/memory/HostRegister.hpp"
#include <complex>
#include <cstdlib>

using dtype = float;

int main(int argc, char *argv[]){

    static_assert( std::is_same<float,dtype>::value, "ONLY FLOAT TYPE IS SUPPORTED DUE TO FFTs" );

    if ( argc != 3 ){
        std::cout << "This program expects 2 user parameters, instead user provided " << argc-1 << std::endl;
        return EXIT_SUCCESS;
    }

    int nSignal = std::stoi(argv[1]);
    int nIters  = std::stoi(argv[2]);

    // allocate host vectors
    ppt::Vector<std::complex<dtype>,ppt::MemSpaceHost> h_vec1(nSignal);
    ppt::Vector<std::complex<dtype>,ppt::MemSpaceHost> h_vec2(nSignal);

    // set initial values to all elements in each vector
    h_vec1.fill(std::complex<dtype>(1,14));
    h_vec2.fill(std::complex<dtype>(1,-1));

    // allocate device vectors
    ppt::Vector<std::complex<dtype>,ppt::MemSpaceCuda> d_vec1(nSignal);
    ppt::Vector<std::complex<dtype>,ppt::MemSpaceCuda> d_vec2(nSignal);

    // create streams
    ppt::StreamCuda::type *stream1 = nullptr;
    ppt::StreamCuda::type *stream2 = nullptr;
    ppt::StreamCuda::create(&stream1);
    ppt::StreamCuda::create(&stream2);

    // create 1d fft plans
    ppt::FourierCuda::type *plan1 = nullptr;
    ppt::FourierCuda::type *plan2 = nullptr;
    ppt::FourierCuda::fftplan1d_float(&plan1, d_vec1.get_nElems(), 1);
    ppt::FourierCuda::fftplan1d_float(&plan2, d_vec2.get_nElems(), 1);

    // attach fft plans to streams
    ppt::FourierCuda::fftplan_attach_stream(plan1,stream1);
    ppt::FourierCuda::fftplan_attach_stream(plan2,stream2);

    // mark host-vector's memory as pinned (non-pageable)
    ppt::HostRegister::mem_lock((void*)h_vec1.get_ptr(), h_vec1.get_nElems()*sizeof(std::complex<dtype>));
    ppt::HostRegister::mem_lock((void*)h_vec2.get_ptr(), h_vec2.get_nElems()*sizeof(std::complex<dtype>));

    // iterate over: H2D, FFT(forward/backward/scale), D2H per stream
    for (int i = 0; i < nIters; i++) {
        ppt::MemSpaceCuda::copyAsyncFromHost(d_vec1.get_ptr(), h_vec1.get_ptr(), h_vec1.get_nElems(), *stream1 );
        ppt::MemSpaceCuda::copyAsyncFromHost(d_vec2.get_ptr(), h_vec2.get_ptr(), h_vec2.get_nElems(), *stream2 );
        
        ppt::FourierCuda::fftplan_exec_forward(plan1,(dtype*)d_vec1.get_ptr(), (dtype*)d_vec1.get_ptr());
        ppt::FourierCuda::fftplan_exec_forward(plan2,(dtype*)d_vec2.get_ptr(), (dtype*)d_vec2.get_ptr());
        
        ppt::FourierCuda::fftplan_exec_backward(plan1,(dtype*)d_vec1.get_ptr(), (dtype*)d_vec1.get_ptr());
        ppt::FourierCuda::fftplan_exec_backward(plan2,(dtype*)d_vec2.get_ptr(), (dtype*)d_vec2.get_ptr());
        
        ppt::FourierCuda::fft_scale(2*d_vec1.get_nElems(), (dtype*)d_vec1.get_ptr(), (dtype)1.0f/d_vec1.get_nElems(), *stream1);
        ppt::FourierCuda::fft_scale(2*d_vec2.get_nElems(), (dtype*)d_vec2.get_ptr(), (dtype)1.0f/d_vec2.get_nElems(), *stream2);

        ppt::MemSpaceCuda::copyAsyncToHost(h_vec1.get_ptr(), d_vec1.get_ptr(), d_vec1.get_nElems(), *stream1 );
        ppt::MemSpaceCuda::copyAsyncToHost(h_vec2.get_ptr(), d_vec2.get_ptr(), d_vec2.get_nElems(), *stream2 );
    }

    // sync streams
    ppt::StreamCuda::sync(stream1);
    ppt::StreamCuda::sync(stream2);

    // check and report max difference with respect to initial values
    dtype max_diff1 = 0;
    dtype max_diff2 = 0;
    for ( int i(0); i < nSignal; i += (int)(nSignal/10) ){
        dtype diff = std::abs( h_vec1[1] - std::complex<dtype>(1,14) );
        if ( diff > max_diff1 )
            max_diff1 = diff;
        diff = std::abs( h_vec2[1] - std::complex<dtype>(1,-1) );
        if ( diff > max_diff2 )
            max_diff2 = diff;
    }
    std::cout << "max_diff1: " << max_diff1 << std::endl;
    std::cout << "max_diff2: " << max_diff2 << std::endl;

    // mark host-vector's memory as pageable
    ppt::HostRegister::mem_unlock((void*)h_vec1.get_ptr(), h_vec1.get_nElems()*sizeof(std::complex<dtype>));
    ppt::HostRegister::mem_unlock((void*)h_vec2.get_ptr(), h_vec2.get_nElems()*sizeof(std::complex<dtype>));

    // destroy ffts plans
    ppt::FourierCuda::fftplan_destroy(plan2);
    ppt::FourierCuda::fftplan_destroy(plan1);

    // destroy streams
    ppt::StreamCuda::destroy(stream1);
    ppt::StreamCuda::destroy(stream2);

    return EXIT_SUCCESS;
}