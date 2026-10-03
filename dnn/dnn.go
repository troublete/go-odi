package dnn

import (
	"math/rand/v2"
)

type Network []float64

func (n Network) Forward(in ...float64) []float64 {
	nlayers := n[0]

	lstart := 2 // 0 is network layer count; 1 is input layer size
	nprev := int(n[1])
	prev := in

	var lsize int
	var nsize int
	var lend int
	var next []float64

	for range int(nlayers - 1) { // start at first hidden layer
		lsize = int(n[lstart])
		lstart = (lstart + 1)           // skip layer size
		nsize = (nprev + 1)             // neuron size (input weights + bias)
		lend = lstart + (nsize * lsize) // layer start + neuron size * layer size

		next = []float64{}
		var idx int
		for idx = lstart; idx < lend; idx += nsize {
			v := n[idx] // bias
			for nidx := range nprev {
				v += prev[nidx] * n[idx+nidx+1] // skip bias, take weights
			}
			next = append(next, Sigmoid(v))
		}

		lstart = lend
		nprev = lsize
		prev = next
	}

	return next
}

// network layout {nlayers, layersizen, layerneuronbiasn, layerneuroninputxn, ...}
func CreateNetwork(layers ...int) Network {
	n := Network{}
	n = append(n, float64(len(layers))) // append layer count as first number
	var last float64
	for _, l := range layers {
		n = append(n, float64(l)) // append n layer size
		if last != 0 {
			for range l {
				n = append(n, rand.Float64()) // append bias
				for range int(last) {
					n = append(n, rand.Float64()) // append input weight
				}
			}
		}
		last = float64(l)
	}
	return n
}
