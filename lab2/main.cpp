#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <tuple>
#include <cmath>
#include "cl.hpp"

#define WORKGROUP_SIZE 256
#define INPUT_FILE "input.txt"
#define OUTPUT_FILE "output.txt"


std::tuple<cl::Context, std::vector<cl::Device>>
init_open_cl()
{
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

    return std::make_tuple(context, devices);
}

cl::Program
load_program(const std::string &filename,
             const cl::Context &context,
             const std::vector<cl::Device> &devices)
{
    std::ifstream program_in(filename);
    std::string program_code(std::istreambuf_iterator<char>(program_in),
                            (std::istreambuf_iterator<char>()));
    cl::Program::Sources program_source(1, std::make_pair(program_code.c_str(), program_code.length() + 1));
    cl::Program program(context, program_source);
    program.build(devices);
    return program;
}

typedef cl::make_kernel<cl::Buffer&, int, int> cl_fn;

void
exec_fn(cl_fn &fn,
        cl::Buffer &buf, size_t n, size_t offset,
        size_t global_size,
        std::vector<cl::Event> &events,
        cl::CommandQueue &queue)
{
    cl::NDRange global_range(global_size);
    cl::NDRange local_range(WORKGROUP_SIZE);
    cl::EnqueueArgs args(queue, events, global_range, local_range);
    events.push_back(fn(args, buf, n, offset));
}

int main()
{
    try {
        cl::Context context;
        std::vector<cl::Device> devices;
        std::tie(context, devices) = init_open_cl();
        cl::CommandQueue queue(context, devices[0]);
        cl::Program program = load_program("program.cl", context, devices);
        cl_fn reduce_fn(program, "do_reduce");
        cl_fn sweep_fn(program, "do_sweep");

        std::ifstream in(INPUT_FILE);
        size_t n, npow2;
        in >> n;
        npow2 = pow(2.0, ceil(log2(n)));
        std::vector<float> in_array(npow2);
        for (size_t i = 0; i < n; ++i)
            in >> in_array[i];

        cl::Buffer out_buf(context, std::begin(in_array), std::end(in_array), false);
        std::vector<cl::Event> events;

        for (size_t offset = 1; npow2 / (offset * 2) >= WORKGROUP_SIZE; offset *= 2)
            exec_fn(reduce_fn, out_buf, npow2, offset, npow2 / offset, events, queue);

        if (npow2 < 512)
            exec_fn(reduce_fn, out_buf, npow2, 1, WORKGROUP_SIZE, events, queue);

        exec_fn(sweep_fn, out_buf, npow2, npow2 / 2, WORKGROUP_SIZE, events, queue);

        for (size_t offset = npow2 / 1024; offset > 0; offset /= 2)
            exec_fn(sweep_fn, out_buf, npow2, offset, npow2 / offset, events, queue);

        std::vector<float> out_array(n);
        queue.enqueueReadBuffer(out_buf, CL_TRUE, 0, sizeof(float) * n, &out_array[0]);

        std::ofstream out(OUTPUT_FILE);
        out << std::fixed << std::setprecision(3);
        for (size_t i = 0; i < n; i++)
            out << out_array[i] << " ";
        out << std::endl;
    }
    catch (cl::Error &e) {
        std::cerr << "ERROR: " << e.what() << " (" << e.err() << ")" << std::endl;
    }
    catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
