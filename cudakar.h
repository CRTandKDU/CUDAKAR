#pragma once
#include <vector>

typedef float karfloat, *karfloat_ptr;

// Edges types and macros for piecewise linear funcs
typedef struct {
	// Number of f-points
	int n_f;
	// Size of f-intervals
	karfloat delta;
	// Range of piecewise linear edge
	karfloat xmin, xmax;
	// Array of f-points
	karfloat_ptr f;
} karedge, *karedge_ptr;

#define KAREDGEP_INDEXOF(x)	(int)std::floor(((x) - edgep->xmin) / edgep->delta)

// Target func types and macros for sampling datasets
typedef karfloat target_func(int n, karfloat_ptr);
typedef void dataset_func(std::vector<karfloat>, void*);

#define DS_TRAINFN "C:\\Users\\chauv\\Documents\\ds_train.dat"

// Nodes types and macros (sum only)
#define KAR_INNERNODE 0
#define KAR_OUTERNODE 1

int gIDNODE = 0;

typedef struct karnode_struct {
	// Type INNER or OUTER
	short kartype;
	int id;
	// Number of addends, i.e. input edges
	int n_edges;
	// Array of addends
	karedge_ptr* edges;
	karnode_struct** subnodes;
} karnode, *karnode_ptr;

// Forward declarations
void kardataset_sample(int n, target_func* target, int sample_size, const char* fn);
karfloat kardataset_loss_mse(karnode_ptr out, int sample_size, karfloat_ptr* xargs, karfloat_ptr y);
void kardataset_stream_map(dataset_func* func, const char* fn, void* clientdata);

karedge_ptr karedge_new(int npoints, karfloat xmin, karfloat xmax);
void karnode_delete(karnode_ptr nodep);
void karedge_rescale(karedge_ptr edgep, karfloat x);
void karedge_print(const char* title, karedge_ptr edgep);
karfloat karedge_eval(karedge_ptr edgep, karfloat x);
void karedge_update(karedge_ptr edgep, karfloat x, karfloat d);

karnode_ptr karnode_new(int n, short kt);
void karnode_delete(karnode_ptr nodep);
void karnode_print(const char* title, karnode_ptr nodep);
karfloat karnode_eval(karnode_ptr nodep, karfloat* xargs);
void karnode_update(karnode_ptr nodep, karfloat* xargs, karfloat y);