package dnn

import (
	"crypto/sha3"
	"math/rand/v2"
	"testing"
)

func TestCreateNetwork(t *testing.T) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 2, 1)
	if n[0] != 0 || // network flags
		n[1] != 3 || // total layer count
		n[2] != 2 || // input layer size
		n[3] != 0 || // input layer activation
		n[4] != 2 || // layer 1 size
		n[5] != 0 || // layer 1 activation
		n[12] != 1 || // output layer size
		n[13] != 0 { // output layer activation
		t.Errorf("failed to build correct format")
	}
}

func TestNetworkRecording(t *testing.T) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 2, 1)
	n.SetRecording()
	if int(n[0])&1 == 0 {
		t.Error("failed to activate recording")
	}
	_, record, err := n.Forward(1, 0)
	if record == nil || err != nil {
		t.Errorf("failed to record")
	}
}

func TestNetworkForward(t *testing.T) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 2, 1)
	result, _, err := n.Forward(0.5, 0.3)
	if len(result) == 0 || err != nil {
		t.Error(err)
		t.Error("failed to propagate")
	}
}

func TestNetworkAnalyse(t *testing.T) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 2, 1)
	meta := n.Analyse()
	if meta.LayerCount != 3 {
		t.Error("failed to extract correct layer count")
	}
	if meta.LayerSizes[0] != 2 || meta.LayerSizes[1] != 2 || meta.LayerSizes[2] != 1 {
		t.Error("failed to extract correct layer sizes")
	}
	if meta.LayerIndex[0] != 2 || meta.LayerIndex[1] != 4 || meta.LayerIndex[2] != 12 {
		t.Error("failed to extract correct layer index")
	}
}

func TestNetworkSetActivation(t *testing.T) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 2, 1)
	n.SetActivation(ACTIVATIONSIGMOID, ACTIVATIONRELU)
	meta := n.Analyse()
	if n[meta.LayerIndex[0]+1] != ACTIVATIONSIGMOID ||
		n[meta.LayerIndex[1]+1] != ACTIVATIONRELU ||
		n[meta.LayerIndex[2]+1] != ACTIVATIONRELU {
		t.Error("failed to set correct layer activations")
	}
}

func TestNetworkBackpropagate(t *testing.T) {
	r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
	n := CreateNetwork(r, 2, 2, 1)
	n.SetRecording()

	var before []float64
	copy(n, before)

	for range 1 {
		err := n.Backpropagate([]float64{1, 0}, []float64{1}, .1)
		if err != nil {
			t.Error(err)
		}
	}

	var after []float64
	copy(n, after)

	bf := []float64{0, 3, 2, 0, 2, 0, 0.9379143012631059, 0.3366480410648668, 0.20347583513135614, 0.6611941052017176, 0.37313416563132307, 0.7258724537161065, 1, 0, 0.43036139413008245, 0.5388613128715701, 0.724286260687862}
	for k := range before {
		if before[k] != bf[k] {
			t.Error("failed to produce expected network")
		}
	}

	af := []float64{1, 3, 2, 0, 2, 0, 0.9382089132691908, 0.3369426530709516, 0.20347583513135614, 0.6616428035257162, 0.3735828639553217, 0.7258724537161065, 1, 0, 0.4335623408692641, 0.5413632582231247, 0.7266481401616498}
	for k := range after {
		if before[k] != af[k] {
			t.Error("failed to produce expected network")
		}
	}
}
