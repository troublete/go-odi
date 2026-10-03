package dnn

import (
	"fmt"
	"testing"
)

func TestCreateNetwork(t *testing.T) {
	n := CreateNetwork(2, 2, 1)
	out := n.Forward(0.5, 0.4)
	fmt.Println(out)
}

var Result []float64

func BenchmarkFF(b *testing.B) {
	n := CreateNetwork(2, 5_000, 5_000, 5_000, 5_000, 5_000, 1_000)
	for i := 0; i < b.N; i++ {
		Result = n.Forward(0.5, 0.4)
	}
}
