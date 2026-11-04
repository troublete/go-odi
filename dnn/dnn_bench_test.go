package dnn

import (
	"crypto/sha3"
	"math/rand/v2"
	"testing"
)

var Result []float64

func BenchmarkFF100M(b *testing.B) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 5_000, 5_000, 5_000, 5_000, 5_000, 1_000)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Result, _, _ = n.Forward(0.5, 0.4)
	}
}

func BenchmarkFF100MRelu(b *testing.B) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 5_000, 5_000, 5_000, 5_000, 5_000, 1_000)
	n.SetActivation(ActivationRelu)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Result, _, _ = n.Forward(0.5, 0.4)
	}
}

func BenchmarkFF10M(b *testing.B) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 3160, 3160, 1)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		Result, _, _ = n.Forward(0.5, 0.4)
	}
}
func BenchmarkFF100K(b *testing.B) {
	b.Run("wide", func(b *testing.B) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n1 := CreateNetwork(r, 2, 316, 316, 1)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Result, _, _ = n1.Forward(0.5, 0.4)
		}
	})

	b.Run("deep", func(b *testing.B) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n2 := CreateNetwork(r, 2, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Result, _, _ = n2.Forward(0.5, 0.4)
		}
	})
}
func BenchmarkFF2M(b *testing.B) {
	b.Run("wide", func(b *testing.B) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n1 := CreateNetwork(r, 2, 1414, 1414, 1)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Result, _, _ = n1.Forward(0.5, 0.4)
		}
	})

	b.Run("deep", func(b *testing.B) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n2 := CreateNetwork(r, 2, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 1)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			Result, _, _ = n2.Forward(0.5, 0.4)
		}
	})
}

func BenchmarkFF2MBackprop(b *testing.B) {
	b.Run("wide", func(b *testing.B) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n1 := CreateNetwork(r, 2, 1414, 1414, 1)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = n1.Backpropagate([]float64{1, 0}, []float64{1}, .1)
		}
	})

	b.Run("deep", func(b *testing.B) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n2 := CreateNetwork(r, 2, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 316, 1)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = n2.Backpropagate([]float64{1, 0}, []float64{1}, .1)
		}
	})
}
