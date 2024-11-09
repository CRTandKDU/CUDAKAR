# CUDAKAR
An exploration of the Kolomogorov-Arnold Representation (KAR) theorem. See e.g.: [Wikipedia](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem)

## Architecture
This repo consists of:
- A KAR server written in C++17 for the [simplest Kolmogorov-Arnold Network](#the-simplest-KAN) (KAN)
- A GUI front-end in Javascript for data visualization

## Tools and Libraries
The HTTP server depends on the single-file, header-only C++ library [cpp-httplib](https://github.com/yhirose/cpp-httplib) by yhirose.

Web pages are rendered through the [inja](https://github.com/pantor/inja) templating engine by pantor  (Version 3.2 only), and dependences therein - with special mention of [json](https://github.com/nlohmann/json) by nlohmann.

Data visualization (line charts mostly) is done with [d3.js](https://d3js.org/), v7.

## The simplest KAN {#the-simplest-KAN}
This KAN represents a target monovariate real function $`y=f(x)`$ by the sum of three intermediate KAN monovariate node outputs, with all edges functions being piecewise linear.

$$ f(x) = \sum_{k=1}^3 \phi_k ( f_k(x) ) $$

All $\phi_k$ and $f_k$ are piecewise linear on the same given interval.

The $\phi_k$ are the KAN *outer functions*; the $f_k$ are the *inner functions*.


