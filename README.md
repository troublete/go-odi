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
BenchmarkFF100M-8         	       8	 133940344 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100M-8         	       8	 135419364 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100M-8         	       8	 133757922 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100M-8         	       8	 133548260 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100M-8         	       8	 133576021 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100MRelu
BenchmarkFF100MRelu-8     	       8	 132232984 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100MRelu-8     	       8	 132052995 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100MRelu-8     	       8	 131462516 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100MRelu-8     	       8	 131548708 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF100MRelu-8     	       8	 133954932 ns/op	  212992 B/op	       6 allocs/op
BenchmarkFF10M
BenchmarkFF10M-8          	      87	  12985376 ns/op	   54536 B/op	       3 allocs/op
BenchmarkFF10M-8          	      92	  12934183 ns/op	   54536 B/op	       3 allocs/op
BenchmarkFF10M-8          	      92	  12933334 ns/op	   54536 B/op	       3 allocs/op
BenchmarkFF10M-8          	      92	  12926046 ns/op	   54536 B/op	       3 allocs/op
BenchmarkFF10M-8          	      92	  13006395 ns/op	   54536 B/op	       3 allocs/op
BenchmarkFF100K
BenchmarkFF100K/wide
BenchmarkFF100K/wide-8    	    7315	    162266 ns/op	    5384 B/op	       3 allocs/op
BenchmarkFF100K/wide-8    	    7342	    162098 ns/op	    5384 B/op	       3 allocs/op
BenchmarkFF100K/wide-8    	    7383	    161764 ns/op	    5384 B/op	       3 allocs/op
BenchmarkFF100K/wide-8    	    7375	    161889 ns/op	    5384 B/op	       3 allocs/op
BenchmarkFF100K/wide-8    	    7431	    161832 ns/op	    5384 B/op	       3 allocs/op
BenchmarkFF100K/deep
BenchmarkFF100K/deep-8    	    6960	    170068 ns/op	    8968 B/op	      11 allocs/op
BenchmarkFF100K/deep-8    	    6781	    169763 ns/op	    8968 B/op	      11 allocs/op
BenchmarkFF100K/deep-8    	    6936	    169532 ns/op	    8968 B/op	      11 allocs/op
BenchmarkFF100K/deep-8    	    7076	    169476 ns/op	    8968 B/op	      11 allocs/op
BenchmarkFF100K/deep-8    	    7082	    169587 ns/op	    8968 B/op	      11 allocs/op
BenchmarkFF2M
BenchmarkFF2M/wide
BenchmarkFF2M/wide-8      	     441	   2685614 ns/op	   24584 B/op	       3 allocs/op
BenchmarkFF2M/wide-8      	     441	   2692952 ns/op	   24584 B/op	       3 allocs/op
BenchmarkFF2M/wide-8      	     445	   2688066 ns/op	   24584 B/op	       3 allocs/op
BenchmarkFF2M/wide-8      	     445	   2688703 ns/op	   24584 B/op	       3 allocs/op
BenchmarkFF2M/wide-8      	     444	   2688951 ns/op	   24584 B/op	       3 allocs/op
BenchmarkFF2M/deep
BenchmarkFF2M/deep-8      	     430	   2770504 ns/op	   53768 B/op	      21 allocs/op
BenchmarkFF2M/deep-8      	     421	   2794279 ns/op	   53768 B/op	      21 allocs/op
BenchmarkFF2M/deep-8      	     429	   2784456 ns/op	   53768 B/op	      21 allocs/op
BenchmarkFF2M/deep-8      	     429	   2786887 ns/op	   53768 B/op	      21 allocs/op
BenchmarkFF2M/deep-8      	     422	   2798867 ns/op	   53768 B/op	      21 allocs/op
BenchmarkFF2MBackprop
BenchmarkFF2MBackprop/wide
BenchmarkFF2MBackprop/wide-8         	      72	  15383609 ns/op	98142822 B/op	      76 allocs/op
BenchmarkFF2MBackprop/wide-8         	      75	  15187830 ns/op	98142823 B/op	      76 allocs/op
BenchmarkFF2MBackprop/wide-8         	      78	  15276011 ns/op	98142823 B/op	      76 allocs/op
BenchmarkFF2MBackprop/wide-8         	      72	  15427685 ns/op	98142809 B/op	      76 allocs/op
BenchmarkFF2MBackprop/wide-8         	      78	  15321975 ns/op	98142883 B/op	      76 allocs/op
BenchmarkFF2MBackprop/deep
BenchmarkFF2MBackprop/deep-8         	      76	  16823246 ns/op	97467495 B/op	     292 allocs/op
BenchmarkFF2MBackprop/deep-8         	      79	  14840372 ns/op	97467478 B/op	     292 allocs/op
BenchmarkFF2MBackprop/deep-8         	      80	  14733608 ns/op	97467505 B/op	     292 allocs/op
BenchmarkFF2MBackprop/deep-8         	      78	  14791821 ns/op	97467474 B/op	     292 allocs/op
BenchmarkFF2MBackprop/deep-8         	      80	  14792402 ns/op	97467493 B/op	     292 allocs/op
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
    Index n+2 = Layer activation function (every layer can choose its own activation function; following options are supported 0 = Sigmoid, 1 = ReLU, 2 = Identity, 3 = Softmax)
    Index n+3-m = Bias + Input weights of neurons in order, sequentially listed 
```

## License

All rights reserved. 
No AI consumption in any form granted.