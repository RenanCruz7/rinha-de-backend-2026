package fraud

import (
	"context"
	"testing"
)

func TestIVFNeighborSearcherFindKNearest(t *testing.T) {
	records := []referenceRecord{
		{Vector: vectorWithDim0(0.01), Label: "legit"},
		{Vector: vectorWithDim0(0.02), Label: "legit"},
		{Vector: vectorWithDim0(0.03), Label: "legit"},
		{Vector: vectorWithDim0(0.80), Label: "fraud"},
		{Vector: vectorWithDim0(0.81), Label: "fraud"},
		{Vector: vectorWithDim0(0.82), Label: "fraud"},
	}

	dataset, err := buildReferenceDataset(records)
	if err != nil {
		t.Fatalf("build dataset failed: %v", err)
	}

	searcher, err := NewIVFNeighborSearcher(dataset, IVFConfig{
		Centroids:     2,
		NProbe:        1,
		MinCandidates: 1,
		SampleSize:    6,
	})
	if err != nil {
		t.Fatalf("create ivf searcher failed: %v", err)
	}

	got, err := searcher.FindKNearest(context.Background(), vectorWithDim0(0.805), 3)
	if err != nil {
		t.Fatalf("find nearest failed: %v", err)
	}

	if len(got) != 3 {
		t.Fatalf("expected 3 neighbors, got %d", len(got))
	}
	if !got[0].IsFraud || !got[1].IsFraud || !got[2].IsFraud {
		t.Fatalf("expected fraud neighbors, got %+v", got)
	}
}

func TestIVFNeighborSearcherFallbackToBruteForce(t *testing.T) {
	records := []referenceRecord{
		{Vector: vectorWithDim0(0.10), Label: "legit"},
		{Vector: vectorWithDim0(0.20), Label: "legit"},
		{Vector: vectorWithDim0(0.30), Label: "legit"},
		{Vector: vectorWithDim0(0.90), Label: "fraud"},
	}

	dataset, err := buildReferenceDataset(records)
	if err != nil {
		t.Fatalf("build dataset failed: %v", err)
	}

	searcher, err := NewIVFNeighborSearcher(dataset, IVFConfig{
		Centroids:     2,
		NProbe:        1,
		MinCandidates: 999, // force fallback
		SampleSize:    4,
	})
	if err != nil {
		t.Fatalf("create ivf searcher failed: %v", err)
	}

	got, err := searcher.FindKNearest(context.Background(), vectorWithDim0(0.89), 2)
	if err != nil {
		t.Fatalf("find nearest failed: %v", err)
	}

	if len(got) != 2 {
		t.Fatalf("expected 2 neighbors, got %d", len(got))
	}
	if !got[0].IsFraud {
		t.Fatalf("expected first neighbor to be fraud, got %+v", got[0])
	}
}
