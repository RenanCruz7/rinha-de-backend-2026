package fraud

import (
	"context"
	"errors"
	"strings"
)

type BootstrapNeighborSearcher struct{}

func NewBootstrapNeighborSearcher() *BootstrapNeighborSearcher {
	return &BootstrapNeighborSearcher{}
}

func (s *BootstrapNeighborSearcher) FindKNearest(_ context.Context, _ []float64, k int) ([]LabeledVector, error) {
	if k <= 0 {
		return nil, errors.New("k must be > 0")
	}

	neighbors := make([]LabeledVector, k)
	for i := range neighbors {
		neighbors[i] = LabeledVector{Label: "legit"}
	}

	return neighbors, nil
}

type ThresholdDecisionPolicy struct{}

func NewThresholdDecisionPolicy() *ThresholdDecisionPolicy {
	return &ThresholdDecisionPolicy{}
}

func (p *ThresholdDecisionPolicy) Decide(neighbors []LabeledVector) (FraudScoreResponse, error) {
	if len(neighbors) != KNearestNeighbors {
		return FraudScoreResponse{}, errors.New("exactly 5 neighbors are required")
	}

	fraudCount := 0
	for _, neighbor := range neighbors {
		if strings.EqualFold(strings.TrimSpace(neighbor.Label), "fraud") {
			fraudCount++
		}
	}

	score := float64(fraudCount) / float64(KNearestNeighbors)
	return FraudScoreResponse{
		Approved:   score < FraudThreshold,
		FraudScore: score,
	}, nil
}
