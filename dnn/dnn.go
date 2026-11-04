package dnn

import (
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	BITRECORD int = 1

	ACTIVATIONSIGMOID float64 = iota
	ACTIVATIONRELU
	ACTIVATIONIDENTITY
)

var (
	ErrInputTooSmall       = fmt.Errorf("input is to small; n must match exact number of input params")
	ErrUnknownActivation   = fmt.Errorf("unknown activation function")
	ErrInvalidTrainingGoal = fmt.Errorf("training goal and output layer size mismatch")
)

type Network []float64

// Inference; see Forward
func (n Network) Inference(in ...float64) ([]float64, []float64, error) {
	return n.Forward(in...)
}

// Forward is the FF function of the network; a.k.a. Inference; when network is
// set to record; the second parameter returned will be the neuron input/outputs per layer (in order)
// this should be used when querying the network
func (n Network) Forward(in ...float64) ([]float64, []float64, error) {
	netflags := int(n[0])
	record := netflags&BITRECORD == 1
	nlayers := n[1]

	lstart := 4 // 0 is network meta info; 1 is network layer count; 2 is input layer size; 3 is layer placeholder for activation function
	nprev := int(n[2])
	prev := in

	if len(in) != nprev {
		return nil, nil, ErrInputTooSmall
	}

	var lsize int
	var lend int
	var lact float64
	var nesize int
	var nevalue float64

	var next []float64
	var inout []float64

	var lit int // layer iterator
	var nit int // neuron iterator

	for range int(nlayers - 1) { // start at first hidden layer
		lsize = int(n[lstart])
		lstart = (lstart + 1 + 1)        // skip layer size + function selection
		nesize = (nprev + 1)             // neuron size (input weights + bias)
		lend = lstart + (nesize * lsize) // layer start + neuron size * layer size
		lact = n[lstart-1]               // activation function for layer

		next = make([]float64, 0, lsize)
		for lit = lstart; lit < lend; lit += nesize {
			nevalue = n[lit] // bias

			for nit = range nprev {
				if record {
					inout = append(inout, prev[nit]) // record inputs
				}
				nevalue += prev[nit] * n[lit+nit+1] // skip bias, take weights
			}

			switch lact {
			case ACTIVATIONSIGMOID:
				nevalue = Sigmoid(nevalue)
			case ACTIVATIONRELU:
				nevalue = Relu(nevalue)
			case ACTIVATIONIDENTITY:
				nevalue = Identity(nevalue)
			default:
				return nil, nil, ErrUnknownActivation
			}

			if record {
				inout = append(inout, nevalue) // record outputs
			}
			next = append(next, nevalue)
		}

		lstart = lend
		nprev = lsize
		prev = next
	}

	return next, inout, nil
}

// SetRecording adds the bit flag as to make the network record its forward pass input & output values
func (n Network) SetRecording() {
	in := int(n[0])
	n[0] = float64(in | BITRECORD)
}

type NetworkMeta struct {
	LayerCount int
	LayerSizes []float64
	LayerIndex []int
}

// Analyse extracts layer meta information from a network
// used e.g. during training to explicitly identify parameters
// to be changed
func (n Network) Analyse() NetworkMeta {
	nm := NetworkMeta{
		LayerCount: int(n[1]),
		LayerSizes: []float64{n[2]},
		LayerIndex: []int{2},
	}

	tridx := 4 // traversal index; skip bitflags + layer count + input layer
	for range nm.LayerCount - 1 {
		lsize := n[tridx]
		lidx := tridx
		// +1 for skipping activation function selection and +1 to actually get to the new layer
		tridx = tridx + 1 + (int(nm.LayerSizes[len(nm.LayerSizes)-1])+1)*int(lsize) + 1
		nm.LayerSizes = append(nm.LayerSizes, lsize)
		nm.LayerIndex = append(nm.LayerIndex, lidx)
	}

	return nm
}

// SetActivation allows to reset the activation function on a network
// to be used after initially creating a new network to update the layer config quickly
// from the default
func (n Network) SetActivation(act ...float64) {
	m := n.Analyse()
	for i, s := range m.LayerIndex {
		n[s+1] = act[int(math.Min(float64(i), float64(len(act)-1)))]
	}
}

// Backpropagate is a simply training function on the network;
// biases and weights are trained with a tanh cap;
// caveat: as pointed out in the README this modules goal is not to be
// substitution for established training frameworks;
// this function is rather to be seen as convenience or descent baseline
// than optimized training
func (n Network) Backpropagate(in []float64, want []float64, learningRate float64) error {
	if int(n[0])&BITRECORD != 1 {
		n.SetRecording()
	}

	m := n.Analyse()

	if len(want) != int(m.LayerSizes[len(m.LayerSizes)-1]) {
		return ErrInvalidTrainingGoal
	}

	out, record, err := n.Forward(in...)
	if err != nil {
		return err
	}

	// calculate output error signal
	oerr := make([]float64, len(out)) // output error signal
	var outact func(float64) float64
	switch n[m.LayerIndex[len(m.LayerIndex)-1]+1] {
	case ACTIVATIONSIGMOID:
		outact = DerivativeSigmoid
	case ACTIVATIONRELU:
		outact = DerivativeRelu
	case ACTIVATIONIDENTITY:
		outact = DerivativeIdentity
	default:
		// unreachable, as the forward pass would already exit
	}
	for i, o := range out {
		oerr[i] = (want[i] - o) * outact(o)
	}

	recidx := len(record)
	changes := make([]float64, len(n))
	for i := len(m.LayerSizes) - 1; i > 0; i-- {
		lsize := m.LayerSizes[i]                 // neuron count
		ninputs := m.LayerSizes[i-1]             // input count
		lidx := m.LayerIndex[i] + 2              // current layer start idx
		recidx = recidx - int(lsize*(1+ninputs)) // record start idx
		plidx := m.LayerIndex[i-1] + 1           // previous layer activation

		var lact func(float64) float64
		switch n[plidx] {
		case ACTIVATIONSIGMOID:
			lact = DerivativeSigmoid
		case ACTIVATIONRELU:
			lact = DerivativeRelu
		case ACTIVATIONIDENTITY:
			lact = DerivativeIdentity
		default:
			// unreachable, as forward pass will already flag it
		}

		properr := make([]float64, int(ninputs))
		for ne := range int(lsize) {
			oer := oerr[ne]

			// add changes to bias
			bidx := lidx + (ne * (int(ninputs) + 1))
			changes[bidx] = changes[bidx] + learningRate*math.Tanh(oer)

			for pi := range int(ninputs) {
				input := record[recidx+pi]
				widx := lidx + (ne * (int(ninputs) + 1)) + pi + 1

				// add changes to the weight
				grad := input * oer
				wchg := math.Tanh(grad) * learningRate
				changes[widx] = changes[widx] + wchg

				properr[pi] = properr[pi] + lact(input)*(oer*n[widx]) // collect propagation error
			}
		}

		oerr = properr // propagate error
	}

	for i, ch := range changes {
		n[i] = n[i] + ch
	}

	return nil
}

// CreateNetwork creates a new network for inference use; activation function set to sigmoid by default
// should be used, for instance, as a bare start for demos, training etc. or to create a format coherent
// baseline
//
//	network layout: []float64{
//	  flags,
//	  nlayers,
//	    layerXsize,
//	    layerXactivation,
//	      layerXneuronYbias,
//	      layerXneuronYinputweightZ,
//	  ...
//	 }
func CreateNetwork(r *rand.Rand, layers ...int) Network {
	n := Network{}
	n = append(n, float64(0))           // set network flags
	n = append(n, float64(len(layers))) // append layer count as first number
	var last float64
	for _, l := range layers {
		n = append(n, float64(l))        // append n layer size
		n = append(n, ACTIVATIONSIGMOID) // append n activation function preference; 0 = sigmoid, 1 = relu
		if last != 0 {
			for range l {
				n = append(n, r.Float64()) // append bias
				for range int(last) {
					n = append(n, r.Float64()) // append input weight
				}
			}
		}
		last = float64(l)
	}
	return n
}
