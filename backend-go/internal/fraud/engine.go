package fraud

import (
	"context"
	"errors"
	"time"
)

const (
	KNearestNeighbors = 5
	FraudThreshold    = 0.6
)

type Engine interface {
	Evaluate(ctx context.Context, req FraudScoreRequest) (FraudScoreResponse, error)
}

type TimedEngine interface {
	Engine
	EvaluateTimed(ctx context.Context, req FraudScoreRequest) (FraudScoreResponse, EvaluationTimings, error)
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
	IsFraud  bool
	Distance float64
}

type EvaluationTimings struct {
	Vectorize time.Duration
	Search    time.Duration
	Decision  time.Duration
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
	resp, _, err := e.EvaluateTimed(ctx, req)
	return resp, err
}

func (e *PipelineEngine) EvaluateTimed(ctx context.Context, req FraudScoreRequest) (FraudScoreResponse, EvaluationTimings, error) {
	var timings EvaluationTimings

	start := time.Now()
	vector, err := e.vectorizer.Vectorize(req)
	if err != nil {
		return FraudScoreResponse{}, timings, err
	}
	timings.Vectorize = time.Since(start)

	start = time.Now()
	neighbors, err := e.searcher.FindKNearest(ctx, vector, KNearestNeighbors)
	if err != nil {
		return FraudScoreResponse{}, timings, err
	}
	timings.Search = time.Since(start)

	start = time.Now()
	resp, err := e.decision.Decide(neighbors)
	timings.Decision = time.Since(start)
	if err != nil {
		return FraudScoreResponse{}, timings, err
	}

	return resp, timings, nil
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

func (e *StubEngine) EvaluateTimed(ctx context.Context, req FraudScoreRequest) (FraudScoreResponse, EvaluationTimings, error) {
	resp, err := e.Evaluate(ctx, req)
	return resp, EvaluationTimings{}, err
}
