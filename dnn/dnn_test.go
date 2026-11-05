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
	t.Run("regular", func(t *testing.T) {
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
	})

	t.Run("multi call idempotency", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetRecording()
		n.SetRecording()
		n.SetRecording()
		n.SetRecording()
		if int(n[0])&1 == 0 {
			t.Error("failed to activate recording")
		}
		_, record, err := n.Forward(1, 0)
		if record == nil || err != nil {
			t.Errorf("failed to record")
		}
	})
}

func TestNetworkForward(t *testing.T) {
	t.Run("regular", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		result, _, err := n.Forward(0.5, 0.3)
		if len(result) == 0 || err != nil {
			t.Error(err)
			t.Error("failed to propagate")
		}
	})

	t.Run("with relu", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetActivation(ActivationRelu)
		result, _, err := n.Forward(0.5, 0.3)
		if len(result) == 0 || err != nil {
			t.Error(err)
			t.Error("failed to propagate")
		}
	})

	t.Run("with identity", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetActivation(ActivationIdentity)
		result, _, err := n.Forward(0.5, 0.3)
		if len(result) == 0 || err != nil {
			t.Error(err)
			t.Error("failed to propagate")
		}
	})

	t.Run("with softmax", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetActivation(ActivationSoftmax)
		result, _, err := n.Forward(0.5, 0.3)
		if len(result) == 0 || err != nil {
			t.Error(err)
			t.Error("failed to propagate")
		}
	})

	t.Run("input too small", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		_, _, err := n.Forward(0.5)
		if err == nil || err != ErrInputTooSmall {
			t.Error("expected error, got none")
		}
	})

	t.Run("unknown activation function", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetActivation(42)
		_, _, err := n.Forward(0.5, 0.6)
		if err == nil || err != ErrUnknownActivation {
			t.Error("expected error, got none")
		}
	})

	t.Run("alias", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		result, _, err := n.Inference(0.5, 0.3)
		if len(result) == 0 || err != nil {
			t.Error(err)
			t.Error("failed to propagate")
		}
	})
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
	n.SetActivation(ActivationSigmoid, ActivationRelu)
	meta := n.Analyse()
	if n[meta.LayerIndex[0]+1] != ActivationSigmoid ||
		n[meta.LayerIndex[1]+1] != ActivationRelu ||
		n[meta.LayerIndex[2]+1] != ActivationRelu {
		t.Error("failed to set correct layer activations")
	}
}

func TestNetworkBackpropagate(t *testing.T) {
	t.Run("regular", func(t *testing.T) {
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
	})

	t.Run("with relu", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetActivation(ActivationRelu)

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

		af := []float64{1, 3, 2, 1, 2, 1, 0.8943510002023033, 0.2930847400040642, 0.20347583513135614, 0.6055636022263029, 0.3175036626559083, 0.7258724537161065, 1, 1, 0.3604112572996504, 0.4586616486588208, 0.652848720333179}
		for k := range after {
			if before[k] != af[k] {
				t.Error("failed to produce expected network")
			}
		}
	})

	t.Run("with identity", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetActivation(ActivationIdentity)

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

		bf := []float64{0, 3, 2, 3, 2, 3, 0.9379143012631059, 0.3366480410648668, 0.20347583513135614, 0.6611941052017176, 0.37313416563132307, 0.7258724537161065, 1, 3, 0.43036139413008245, 0.5388613128715701, 0.724286260687862}
		for k := range before {
			if before[k] != bf[k] {
				t.Error("failed to produce expected network")
			}
		}

		af := []float64{1, 3, 2, 3, 2, 3, 0.8943510002023033, 0.2930847400040642, 0.20347583513135614, 0.6055636022263029, 0.3175036626559083, 0.7258724537161065, 1, 3, 0.3604112572996504, 0.4586616486588208, 0.652848720333179}
		for k := range after {
			if before[k] != af[k] {
				t.Error("failed to produce expected network")
			}
		}
	})

	t.Run("with softmax", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 2)
		n.SetActivation(ActivationSoftmax)

		var before []float64
		copy(n, before)

		for range 1 {
			err := n.Backpropagate([]float64{1, 0}, []float64{1, 0}, .1)
			if err != nil {
				t.Error(err)
			}
		}

		var after []float64
		copy(n, after)

		bf := []float64{1, 3, 2, 3, 2, 3, 0.9379143012631059, 0.3366480410648668, 0.20347583513135614, 0.6611941052017176, 0.37313416563132307, 0.7258724537161065, 2, 3, 0.43036139413008245, 0.5388613128715701, 0.724286260687862, 0.41026470413080574, 0.1996286752644929, 0.4689901708047298}
		for k := range before {
			if before[k] != bf[k] {
				t.Error("failed to produce expected network")
			}
		}

		af := []float64{1, 3, 2, 3, 2, 3, 0.9387832176959388, 0.3375169574976996, 0.20347583513135614, 0.6603251887688848, 0.37226524919849024, 0.7258724537161065, 2, 3, 0.470062550425814, 0.5619530098520816, 0.7425721060966177, 0.37056354783507417, 0.1765369782839814, 0.45070432539597405}
		for k := range after {
			if before[k] != af[k] {
				t.Error("failed to produce expected network")
			}
		}
	})

	t.Run("not triggering recording manually", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)

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
	})

	t.Run("unknown activation input", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)
		n.SetActivation(42)

		var before []float64
		copy(n, before)

		err := n.Backpropagate([]float64{1, 0}, []float64{1}, .1)
		if err == nil || err != ErrUnknownActivation {
			t.Error("expected err, got none")
		}
	})

	t.Run("invalid training goal", func(t *testing.T) {
		r := rand.New(rand.NewChaCha8(sha3.Sum256([]byte("test"))))
		n := CreateNetwork(r, 2, 2, 1)

		err := n.Backpropagate([]float64{1, 0}, []float64{1, 42, 1337}, .1)
		if err == nil || err != ErrInvalidTrainingGoal {
			t.Error("expected err, got none")
		}
	})

}
