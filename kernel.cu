
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
#include <sstream>
#include <string>
#include <vector>
#include <random>

#include "httplib.h"
#include "inja.hpp"

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

void kardataset_stream_map(dataset_func *func, const char* fn, void* clientdata) {
    std::string line, word, temp;
    std::vector<karfloat> row;
    std::ifstream ds_file(fn);
    while (!ds_file.eof()) {
        row.clear();
        std::getline(ds_file, line);
        if (line.rfind("#", 0) == 0) {
            // A comment
        }
        else {
            std::stringstream s(line);
            while (getline(s, word, ','))
            {
                row.push_back((karfloat) std::stof(word));
            }
            func(row, clientdata);
        }
    }
    ds_file.close();
}

karfloat kardataset_loss_mse(karnode_ptr out, int sample_size, karfloat_ptr* xargs, karfloat_ptr y) {
    karfloat loss = 0., d;
    for (int i = 0 ; i < sample_size; i++) {
        d = karnode_eval(out, xargs[i]) - y[i];
        loss += d * d;
    }
    return loss;
}

// -------------------------------------------------------------------------------------
// Node operations
karnode_ptr karnode_new(int n, short kt) {
    karnode_ptr nodep = new karnode;
    nodep->kartype = kt;
    nodep->n_edges = n;
    nodep->id = gIDNODE++;
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
            karfloat zi = karedge_eval(nodep->edges[i], yi);

            if (yi > 100. || yi < -100.) {
                std::cout << "Node eval (1): High res! " << yi << "\t" << std::endl;
                            }
                        
           
           //std::cout << "\tZi: " << yi << "\t" << zi << std::endl;
           res += zi;
        }
    }

    if (res > 100. || res < -100.) {
        std::cout << "Node eval (2): High res! " << res << "\t" << std::endl;
        
    }
    return res;
}

void karnode_print(const char *title, karnode_ptr nodep) {
    char buf[3];
    std::cout << "Node: " << title << "\t" << nodep->id << std::endl;
    for (int i = 0; i < nodep->n_edges; i++) {
        sprintf(buf, "E%d", i);
        karedge_print(buf, nodep->edges[i]);
    }
}

void karnode_update(karnode_ptr nodep, karfloat* xargs, karfloat y) {
    karfloat yest = karnode_eval(nodep, xargs);
    

    karnode_print( "Pre-update ", nodep);

    if (KAR_INNERNODE == nodep->kartype) {
        karfloat d = y / nodep->n_edges;
        for (int i = 0; i < nodep->n_edges; i++) {
            karedge_update(nodep->edges[i], xargs[i], d);
        }
    }

    if (KAR_OUTERNODE == nodep->kartype) {
        karfloat d = (y - yest) / nodep->n_edges;
        for (int i = 0; i < nodep->n_edges; i++) {
            // Update outer edges
            karfloat zi = karnode_eval(nodep->subnodes[i], xargs);
            karedge_rescale(nodep->edges[i], zi);
            karedge_update(nodep->edges[i], zi, d);
            // Update inner edges
            karnode_update(nodep->subnodes[i], xargs, d);
        }
    }

    karnode_print("Post update ", nodep);
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

    std::random_device rd;  // a seed source for the random number engine
    std::mt19937 gen(rd()); // mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(10, 90);

    for (int i = 0; i < npoints; edgep->f[i++] = (karfloat) distrib(gen)/100.);
    return edgep;
}

void karedge_rescale(karedge_ptr edgep, karfloat x) {
    karfloat range;
    bool shift = false;
    if (x < edgep->xmin) {
        edgep->xmin = x; shift = true;
    }
    if (x > edgep->xmax) {
        edgep->xmax = x; shift = true;
    }
    if (shift) {
        range = edgep->xmax - edgep->xmin;
        edgep->xmin -= .01 * range;
        edgep->xmax += .01 * range;
        edgep->delta = (edgep->xmax - edgep->xmin) / (edgep->n_f - 1);
    }
}

void karedge_delete(karedge_ptr edgep) {
    delete edgep->f;
    delete edgep;
}

void karedge_print(const char* title, karedge_ptr edgep) {
    std::cout << title << ": ";
    for (int i = 0; i < edgep->n_f; i++) {
        std::cout << std::fixed << std::setw(5) << std::setprecision(4) << edgep->f[i] << " ";
    }
    std::cout << "[" << edgep->xmin << ", " << edgep->xmax << "]" << std::endl;
}

karfloat karedge_eval(karedge_ptr edgep, karfloat x ) {
    // Adjust if last f-point
    if (x >= edgep->xmax) x = edgep->xmax - 0.000001;
    if (x <= edgep->xmin) x = edgep->xmin + 0.000001;
    int index = KAREDGEP_INDEXOF(x);
    if (index < 0) {
        std::cerr << "Eval: Negative index! " << x << std::endl;
        exit(1);
    }
    karfloat offset = ((x - edgep->xmin) / edgep->delta) - index;
    return edgep->f[index] * (1. - offset) + edgep->f[index + 1] * offset;
}

void karedge_update(karedge_ptr edgep, karfloat x, karfloat d) {
    if (x >= edgep->xmax) x = edgep->xmax - 0.000001;
    if (x <= edgep->xmin) x = edgep->xmin + 0.000001;
    int index = KAREDGEP_INDEXOF(x);
    if (index < 0) {
        std::cerr << "Update: Negative index! " << x << std::endl;
        exit(1);
    }
    karfloat offset = ((x - edgep->xmin) / edgep->delta) - index;
    karfloat est = edgep->f[index] * (1. - offset) + edgep->f[index + 1] * offset;
    karfloat den = (1. - offset) * (1. - offset) + offset * offset;
    den *= gALPHA;
    if (den > 100. || den < -100.) {
        std::cerr << "Update: High den! " << den << "\tOffset: " << offset << std::endl;
    }
    // Standard gradient descent
    std::cout << "Edge update " << index << ": " << d * (1. - offset) / den << ", " << d * offset / den << std::endl;
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

void ds_print(std::vector<karfloat> row, void* ignore) {
    int n = row.size();
    if (n <= 0) return;

    for (int i = 0; i < n - 1; i++) {
        std::cout << std::fixed << row[i] << "\t";
    }
    std::cout << "--> " << row[n - 1] << std::endl;
}

void ds_update(std::vector<karfloat> row, void* clientdata) {
    int n = row.size();
    if (n <= 0) return;

    karnode_ptr out = (karnode_ptr)clientdata;
    karfloat_ptr xargs = new karfloat[n - 1];
    for (int i = 0; i < n - 1; i++) xargs[i] = row[i];
    
    for (int i = 0; i < n ; i++) std::cout << std::fixed << row[i] << "\t";
    std::cout << std::endl;

    karnode_update(out, xargs, row[n - 1]);

    delete xargs;
}

typedef struct {
    karnode_ptr out;
    double cumul;
} mse_clientdata, *mse_clientdata_ptr;

void ds_mse_loss(std::vector<karfloat> row, void* clientdata) {
    int n = row.size();
    if (n <= 0) return;

    mse_clientdata_ptr g = (mse_clientdata_ptr)clientdata;
    karfloat_ptr xargs = new karfloat[n - 1];
    for (int i = 0; i < n - 1; i++) xargs[i] = row[i];

    karfloat yest = karnode_eval(g->out, xargs);
    // std::cout << std::fixed << yest << "\t" << row[n - 1] << "\t" << g->cumul << std::endl;
    g->cumul += (yest - row[n - 1]) * (yest - row[n - 1]);

    delete xargs;
}

inja::json data_to_json( karnode_ptr out ) {
    // JSON-serialize edges current definitions
    inja::json outer_jdata, inner_jdata, info;
    for (int i = 0; i < out->n_edges; i++) {
        inja::json jdata_i;
        for (int j = 0; j < out->edges[i]->n_f; j++) {
            inja::json jdata_ipoint;
            jdata_ipoint["x"] = out->edges[i]->xmin + j * out->edges[i]->delta;
            jdata_ipoint["y"] = out->edges[i]->f[j];
            jdata_i[j] = jdata_ipoint;
        }
        outer_jdata[i] = jdata_i;
    }
    info["outer_fpoints"] = outer_jdata;
    
    for (int i = 0; i < out->n_edges; i++) {
        inja::json jdata_i;
        for (int j = 0; j < out->subnodes[i]->n_edges; j++) {
            inja::json jdata_j;
            karedge_ptr edgep = out->subnodes[i]->edges[j];
            for (int k = 0; k < edgep->n_f; k++) {
                inja::json jdata_kpoint;
                jdata_kpoint["x"] = edgep->xmin + k * edgep->delta;
                jdata_kpoint["y"] = edgep->f[k];
                jdata_j[k] = jdata_kpoint;
            }
            jdata_i[j] = jdata_j;
        }
        inner_jdata[i] = jdata_i;
    }
    info["inner_fpoints"] = inner_jdata;

    // Compute and JSON-serialize MSE loss
    mse_clientdata data = { out, 0. };
    kardataset_stream_map(ds_mse_loss, DS_TRAINFN, (void*)&data);
    info["loss"] = data.cumul;
    return info;
}


int main() {
    // kardataset_sample(1, target_test_1, 30, "C:\\Users\\chauv\\Documents\\ds_train.dat");

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


    //for (int epoch = 0; epoch < 1; epoch++) {
    //    mse_clientdata data = { out, 0. };
    //    kardataset_stream_map(ds_mse_loss, DS_TRAINFN, (void*)&data);
    //    std::cout << "MSE Loss: " << data.cumul << std::endl;
    //    kardataset_stream_map(ds_update, DS_TRAINFN, (void*)out);
    //}
    //mse_clientdata data = { out, 0. };
    //kardataset_stream_map(ds_mse_loss, DS_TRAINFN, (void*)&data);
    //std::cout << "MSE Loss: " << data.cumul << std::endl;

    // HTTP
    httplib::Server svr;

    svr.Get("/hi", [&](const httplib::Request&, httplib::Response& res) {
        inja::json data = data_to_json( out );  
        inja::Environment env;
        data["title"] = "Dashboard";
        inja::Template temp = env.parse_template("templates/dashboard.html");
        std::string resp = env.render(temp, data);

        res.set_content(resp, "text/html");
        });

    svr.Get("/data", [&](const httplib::Request& req, httplib::Response& res) {
        kardataset_stream_map(ds_update, DS_TRAINFN, (void*)out);

        inja::json info = data_to_json(out);
        res.set_content(info.dump(), "application/json");
        });

    svr.Get("/stop", [&](const httplib::Request& req, httplib::Response& res) {
        svr.stop();
        });

    svr.listen("0.0.0.0", 8080);

    delete out;

    return 0;
}