#pragma once
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

// Nodes types and macros (sum only)
#define KAR_INNERNODE 0
#define KAR_OUTERNODE 1

typedef struct karnode_struct {
	// Type INNER or OUTER
	short kartype;
	// Number of addends, i.e. input edges
	int n_edges;
	// Array of addends
	karedge_ptr* edges;
	karnode_struct** subnodes;
} karnode, *karnode_ptr;

// Forward decls
void kardataset_sample(int n, target_func* target, int sample_size, const char* fn);

karedge_ptr karedge_new(int npoints, karfloat xmin, karfloat xmax);
void karnode_delete(karnode_ptr nodep);
karfloat karedge_eval(karedge_ptr edgep, karfloat x);
void karedge_update(karedge_ptr edgep, karfloat x, karfloat d);

karnode_ptr karnode_new(int n, short kt);
void karnode_delete(karnode_ptr nodep);
karfloat karnode_eval(karnode_ptr nodep, karfloat* xargs);
