package dnn

import "math"

func Sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Pow(math.E, -1.0*x))
}

func DerivativeSigmoid(y float64) float64 {
	x := Sigmoid(y)
	return x * (1 - x)
}
