#define __CL_ENABLE_EXCEPTIONS

#if defined(__APPLE__) || defined(__MACOSX)
#include "cl.hpp"
#else
#include <CL/cl.hpp>
#endif

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#define INPUT_FILE "input.txt"
#define OUTPUT_FILE "output.txt"

void read_matrix(std::ifstream &in, std::vector<float> &matrix, size_t size)
{
    matrix.resize(size * size);
    for (size_t i = 0; i < size; i++)
        for (size_t j = 0; j < size; j++)
            in >> matrix[i * size + j];
}

int main()
{
    try {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
            throw std::runtime_error("No suitable platform found");

        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)(platforms[0])(),
            0
        };
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);

        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty())
            throw std::runtime_error("No suitable device found");

        std::ifstream program_in("program.cl");
        std::string program_code(std::istreambuf_iterator<char>(program_in),
                                (std::istreambuf_iterator<char>()));
        cl::Program::Sources program_source(1, std::make_pair(program_code.c_str(), program_code.length() + 1));
        cl::Program program(context, program_source);
        program.build(devices);

        cl::CommandQueue queue(context, devices[0], 0);

        size_t n, m;
        std::vector<float> fst_matrix, snd_matrix;
        std::ifstream in(INPUT_FILE);
        in >> n >> m;
        read_matrix(in, fst_matrix, n);
        read_matrix(in, snd_matrix, m);

        cl::Buffer fst_buf(context, std::begin(fst_matrix),
                                    std::end(fst_matrix), true, false);
        cl::Buffer snd_buf(context, std::begin(snd_matrix),
                                    std::end(snd_matrix), true, false);
        cl::Buffer result_buf(context, CL_MEM_WRITE_ONLY, sizeof(float) * n * n);

        cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, int, int>
            convolution_fn(program, "convolution");

        size_t local_size  = 16;
        size_t global_size = ((n + local_size - 1) / local_size) * local_size; // TODO: revise
        cl::EnqueueArgs args(queue, cl::NDRange(global_size, global_size), cl::NDRange(local_size, local_size));

        convolution_fn(args, fst_buf, snd_buf, result_buf, n, m).wait();

        std::vector<float> result(n*n);
        queue.enqueueReadBuffer(result_buf, CL_TRUE, 0, sizeof(float) * n * n, &result[0]);

        std::ofstream out(OUTPUT_FILE);
        out << std::fixed << std::setprecision(3);
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++)
                out << result[i * n + j] << " ";
            out << std::endl;
        }
    }
    catch (cl::Error &e) {
        std::cerr << "ERROR: " << e.what() << " (" << e.err() << ")" << std::endl;
    }
    catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
