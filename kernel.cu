
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int cuda_main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


#include <cmath>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>

#include "cudakar.h"

karfloat gALPHA = 1.;

// -------------------------------------------------------------------------------------
// Sampling datasets
void kardataset_sample(int n, target_func *target, int sample_size, const char *fn) {
    std::time_t current_time = std::time(nullptr);
    std::tm* local_time = std::localtime(&current_time);
    std::ofstream ds_file(fn);
    ds_file << std::setprecision(3) << "# Gen at: " << std::asctime(local_time);

    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0, 31416);

    karfloat y;
    for (int i = 0; i < sample_size; i++) {
        karfloat_ptr xargs = new karfloat[n];
        for (int j = 0; j < n; j++) {
            xargs[j] = distrib(gen) / 10000.;
            ds_file << std::fixed << xargs[j] << ", ";
        }
        y = target(n, xargs);
        ds_file << std::fixed << y << std::endl;
        delete xargs;
    }
    ds_file.close();
}

// -------------------------------------------------------------------------------------
// Node operations
karnode_ptr karnode_new(int n, short kt) {
    karnode_ptr nodep = new karnode;
    nodep->kartype = kt;
    nodep->n_edges = n;
    karedge_ptr* ep = new karedge_ptr[n];
    nodep->edges = ep;
    if (KAR_OUTERNODE == kt) {
        karnode_ptr* np = new karnode_ptr[n];
        nodep->subnodes = np;
    }
    else nodep->subnodes = (karnode_ptr*)0;

    return nodep;
}

void karnode_delete(karnode_ptr nodep) {
    delete nodep->edges;
    if (nodep->subnodes) delete nodep->subnodes;
    delete nodep;
}

karfloat karnode_eval(karnode_ptr nodep, karfloat* xargs) {
    karfloat res = (karfloat)0.;
    for (int i = 0; i < nodep->n_edges; i++) {
        if (KAR_INNERNODE == nodep->kartype) {
            res += karedge_eval(nodep->edges[i], xargs[i]);
        }
        else {
            karfloat yi = karnode_eval(nodep->subnodes[i], xargs);
            res += karedge_eval(nodep->edges[i], yi);
        }
    }
    return res;
}

// -------------------------------------------------------------------------------------
// Edge operations
karedge_ptr karedge_new(int npoints, karfloat xmin, karfloat xmax) {
    //karedge_ptr edgep = (karedge_ptr)std::malloc(sizeof(karedge));
    //karfloat fp = (karfloat_ptr)std::malloc((size_t)npoints);
    karedge_ptr edgep = new karedge;
    karfloat_ptr fp = new karfloat[npoints];
    edgep->n_f = npoints;
    edgep->xmin = xmin;
    edgep->xmax = xmax;
    edgep->delta = (xmax - xmin) / (npoints - 1);
    edgep->f = fp;
    for (int i = 0; i < npoints; edgep->f[i++] = (karfloat).5);
    return edgep;
}

void karedge_delete(karedge_ptr edgep) {
    delete edgep->f;
    delete edgep;
}

karfloat karedge_eval(karedge_ptr edgep, karfloat x ) {
    // Adjust if last f-point
    if (x >= edgep->xmax) x -= 0.000001;
    int index = KAREDGEP_INDEXOF(x);
    karfloat offset = ((x - edgep->xmin) / edgep->delta) - index;
    return edgep->f[index] * (1. - offset) + edgep->f[index + 1] * offset;
}

void karedge_update(karedge_ptr edgep, karfloat x, karfloat d) {
    if (x >= edgep->xmax) x -= 0.000001;
    int index = KAREDGEP_INDEXOF(x);
    karfloat offset = ((x - edgep->xmin) / edgep->delta) - index;
    karfloat est = edgep->f[index] * (1. - offset) + edgep->f[index + 1] * offset;
    karfloat den = (1. - offset) * (1. - offset) + offset * offset;
    den *= gALPHA;
    // Standard gradient descent
    edgep->f[index] += d * (1. - offset) / den;
    edgep->f[index + 1] += d * offset / den;
}


// -------------------------------------------------------------------------------------
karfloat target_test_2(int n, karfloat_ptr xargs) {
    return std::sin(xargs[0]) + std::cos(xargs[1] * 2.);
}

karfloat target_test_1(int n, karfloat_ptr xargs) {
    return std::sin(xargs[0]) + std::cos(xargs[0] * 2.);
}

int main() {
    kardataset_sample(1, target_test_1, 30, "C:\\Users\\chauv\\Documents\\ds_train.dat");

    // A 1/3/1 KAN
    karedge_ptr* inners = new karedge_ptr[3];
    karedge_ptr* outers = new karedge_ptr[3];
    karnode_ptr* subnodes = new karnode_ptr[3];
    karnode_ptr out;

    for (int i = 0; i < 3; i++) {
        inners[i] = karedge_new(7, 0., 3.14159265);
        outers[i] = karedge_new(11, 0., 3.14159265);
        subnodes[i] = karnode_new(1, KAR_INNERNODE);
        subnodes[i]->edges = &inners[i];
    }
    out = karnode_new(3, KAR_OUTERNODE);
    for (int i = 0; i < 3; i++) {
        out->edges[i] = outers[i];
        out->subnodes[i] = subnodes[i];
    }

    karfloat xargs[1] = { .333 };
    std::cout << std::fixed << karnode_eval(out, xargs) << std::endl;
    
    delete out;

    return 0;
}