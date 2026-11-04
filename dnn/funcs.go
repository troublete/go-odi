package dnn

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -1.0*x))
}

func DerivativeSigmoid(x float64) float64 {
	return x * (1 - x)
}

func Relu(x float64) float64 {
	return math.Max(0, x)
}

func DerivativeRelu(y float64) float64 {
	if y > 0 {
		return 1
	} else {
		return 0
	}
}

func Identity(x float64) float64 {
	return x
}

func DerivativeIdentity(_ float64) float64 {
	return 1
}
