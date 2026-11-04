# go-odi 
> pure go, CPU-only: DNN-format, inference and training

[![test](https://github.com/troublete/go-odi/actions/workflows/test.yml/badge.svg)](https://github.com/troublete/go-odi/actions/workflows/test.yml)


## Introduction

This module is meant to provide efficient, accessible and scalable DNN integration for any application. It is build upon
a 1D slice, so that the memory is as efficiently used as the runtime permits is.

```go
n := NewNetwork(rand.New(rand.NewSource(...)), 3, 4, 16, 2) // creates a network with a input layer size 2, two hidden layer (size 4 and 16) and an output layer 2
result, record, err := n.Forward(.2, .3) // runs a forward pass through the network; with input .2 and .3; the result is a []float64, record allows the retrieval of intermediary input/output values per neuron (used also for backpropagation internally)
// n.Inference(...) is an alias on n.Forward(...)
```

## Scope & Limitations

Though this module provides functions for working with the format and training a network, the focus of this library is
the forward pass; the inference. So the provided backpropagation (training) functionality is to be seen as a convenience
layer and provides a descent baseline, especially for regular DNN uses (i.e. 2-5M parameters) but can't achieve the
performance of industry standard workflows.

Additionally it is to be highlighted that the scope of this library is NOT to be the fastest DNN inference out there.
It is meant to provide the most efficient, idiomatic DNN inference in pure Go; while avoiding custom assembly code,
dependencies and most importantly C bindings for GPU computations.

Further: right now this whole module is fully built on float64.

The benchmarks will show that it solves all of this adequately. 

```txt
cpu: Apple M1
BenchmarkFF100M
BenchmarkFF100M-8         	       7	 148820143 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100MRelu
BenchmarkFF100MRelu-8     	       8	 133111417 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF10M
BenchmarkFF10M-8          	      91	  13019464 ns/op	   54536 B/op	       3 allocs/op
BenchmarkFF100K
BenchmarkFF100K/wide
BenchmarkFF100K/wide-8    	    7192	    163010 ns/op	    5384 B/op	       3 allocs/op
BenchmarkFF100K/deep
BenchmarkFF100K/deep-8    	    6475	    170830 ns/op	    8968 B/op	      11 allocs/op
BenchmarkFF2M
BenchmarkFF2M/wide
BenchmarkFF2M/wide-8      	     441	   2696968 ns/op	   24584 B/op	       3 allocs/op
BenchmarkFF2M/deep
BenchmarkFF2M/deep-8      	     429	   2775034 ns/op	   53768 B/op	      21 allocs/op
BenchmarkFF2MBackprop
BenchmarkFF2MBackprop/wide
BenchmarkFF2MBackprop/wide-8         	      81	  14331912 ns/op	98063836 B/op	      52 allocs/op
BenchmarkFF2MBackprop/deep
BenchmarkFF2MBackprop/deep-8         	      79	  14243823 ns/op	97303838 B/op	      94 allocs/op
```

## Network format

To work efficiently this library is using an approach where all network values are kept in a 1D float64 slice, so that
the memory is as efficiently used as possible, therefore networks created by this module, or which ought to be used with
it must follow a defined format.

The index structure is as follows:

```txt
Index 0 (network[0]) = RESERVED (used for bitflags at the moment to trigger recording of values), should be set to 0
Index 1 = Total layer count (input+hidden+output)
Index 2-n = Layer definitions (the layers are defined in sequential orders)
    Index n+1 = Layer size (the amount of fully connected neurons are on the layer)
    Index n+2 = Layer activation function (every layer can choose its own activation function; following options are supported 0 = Sigmoid, 1 = ReLU)
    Index n+3-m = Bias + Input weights of neurons in order, sequentially listed 
```

## License

All rights reserved. 
No AI consumption in any form granted.