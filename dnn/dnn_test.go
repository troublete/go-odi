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

func BenchmarkFF100M(b *testing.B) {
	n := CreateNetwork(2, 5_000, 5_000, 5_000, 5_000, 5_000, 1_000)
	for i := 0; i < b.N; i++ {
		Result = n.Forward(0.5, 0.4)
	}
}

func BenchmarkFF10M(b *testing.B) {
	n := CreateNetwork(2, 3160, 3160, 1)
	for i := 0; i < b.N; i++ {
		Result = n.Forward(0.5, 0.4)
	}
}

func BenchmarkFFMicro(b *testing.B) {
	b.Run("wide", func(b *testing.B) {
		n1 := CreateNetwork(2, 316, 316, 1)
		for i := 0; i < b.N; i++ {
			Result = n1.Forward(0.5, 0.4)
		}
	})

	b.Run("deep", func(b *testing.B) {
		n2 := CreateNetwork(2, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1)
		for i := 0; i < b.N; i++ {
			Result = n2.Forward(0.5, 0.4)
		}
	})
}

func BenchmarkFFStandard(b *testing.B) {
	b.Run("wide", func(b *testing.B) {
		n1 := CreateNetwork(2, 1414, 1414, 1)
		for i := 0; i < b.N; i++ {
			Result = n1.Forward(0.5, 0.4)
		}
	})

	b.Run("deep", func(b *testing.B) {
		n2 := CreateNetwork(2, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 1)
		for i := 0; i < b.N; i++ {
			Result = n2.Forward(0.5, 0.4)
		}
	})
}
