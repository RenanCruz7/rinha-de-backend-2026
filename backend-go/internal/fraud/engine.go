package fraud

import (
	"context"
	"errors"
)

const (
	KNearestNeighbors = 5
	FraudThreshold    = 0.6
)

type Engine interface {
	Evaluate(ctx context.Context, req FraudScoreRequest) (FraudScoreResponse, error)
}

type Vectorizer interface {
	Vectorize(req FraudScoreRequest) ([]float64, error)
}

type NeighborSearcher interface {
	FindKNearest(ctx context.Context, vector []float64, k int) ([]LabeledVector, error)
}

type DecisionPolicy interface {
	Decide(neighbors []LabeledVector) (FraudScoreResponse, error)
}

type LabeledVector struct {
	Label    string
	Distance float64
}

type PipelineEngine struct {
	vectorizer Vectorizer
	searcher   NeighborSearcher
	decision   DecisionPolicy
}

func NewPipelineEngine(vectorizer Vectorizer, searcher NeighborSearcher, decision DecisionPolicy) (*PipelineEngine, error) {
	if vectorizer == nil || searcher == nil || decision == nil {
		return nil, errors.New("vectorizer, searcher and decision are required")
	}

	return &PipelineEngine{
		vectorizer: vectorizer,
		searcher:   searcher,
		decision:   decision,
	}, nil
}

func (e *PipelineEngine) Evaluate(ctx context.Context, req FraudScoreRequest) (FraudScoreResponse, error) {
	vector, err := e.vectorizer.Vectorize(req)
	if err != nil {
		return FraudScoreResponse{}, err
	}

	neighbors, err := e.searcher.FindKNearest(ctx, vector, KNearestNeighbors)
	if err != nil {
		return FraudScoreResponse{}, err
	}

	return e.decision.Decide(neighbors)
}

type StubEngine struct{}

func NewStubEngine() *StubEngine {
	return &StubEngine{}
}

func (e *StubEngine) Evaluate(_ context.Context, _ FraudScoreRequest) (FraudScoreResponse, error) {
	return FraudScoreResponse{
		Approved:   true,
		FraudScore: 0.0,
	}, nil
}
