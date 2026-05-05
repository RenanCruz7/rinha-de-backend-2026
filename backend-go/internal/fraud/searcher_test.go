package fraud

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestBruteForceNeighborSearcherReturnsFiveNeighborsOrderedByDistance(t *testing.T) {
	records := []referenceRecord{
		{Vector: vectorWithDim0(0.20), Label: "v0"},
		{Vector: vectorWithDim0(0.10), Label: "v1"},
		{Vector: vectorWithDim0(0.30), Label: "v2"},
		{Vector: vectorWithDim0(0.40), Label: "v3"},
		{Vector: vectorWithDim0(0.05), Label: "v4"},
		{Vector: vectorWithDim0(0.90), Label: "v5"},
	}

	dataset, err := buildReferenceDataset(records)
	if err != nil {
		t.Fatalf("build dataset failed: %v", err)
	}

	searcher, err := NewBruteForceNeighborSearcher(dataset)
	if err != nil {
		t.Fatalf("create searcher failed: %v", err)
	}

	query := vectorWithDim0(0.20)
	got, err := searcher.FindKNearest(context.Background(), query, 5)
	if err != nil {
		t.Fatalf("find nearest failed: %v", err)
	}

	if len(got) != 5 {
		t.Fatalf("expected 5 neighbors, got %d", len(got))
	}

	wantLabels := []string{"v0", "v1", "v2", "v4", "v3"}
	for i, want := range wantLabels {
		if got[i].Label != want {
			t.Fatalf("unexpected label at %d: got %q want %q", i, got[i].Label, want)
		}
	}

	for i := 1; i < len(got); i++ {
		if got[i].Distance < got[i-1].Distance {
			t.Fatalf("neighbors not sorted by ascending distance at index %d", i)
		}
	}
}

func TestBruteForceNeighborSearcherTieBreakByDatasetOrder(t *testing.T) {
	records := []referenceRecord{
		{Vector: vectorWithDim0(0.10), Label: "first"},
		{Vector: vectorWithDim0(0.30), Label: "second"},
	}

	dataset, err := buildReferenceDataset(records)
	if err != nil {
		t.Fatalf("build dataset failed: %v", err)
	}

	searcher, err := NewBruteForceNeighborSearcher(dataset)
	if err != nil {
		t.Fatalf("create searcher failed: %v", err)
	}

	query := vectorWithDim0(0.20)
	got, err := searcher.FindKNearest(context.Background(), query, 2)
	if err != nil {
		t.Fatalf("find nearest failed: %v", err)
	}

	if got[0].Label != "first" || got[1].Label != "second" {
		t.Fatalf("unexpected tie order: %+v", got)
	}
}

func TestLoadReferenceDatasetFromGzip(t *testing.T) {
	records := []referenceRecord{
		{Vector: vectorWithDim0(0.10), Label: "legit"},
		{Vector: vectorWithDim0(0.90), Label: "fraud"},
	}

	path := filepath.Join(t.TempDir(), "references.json.gz")
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("create temp file failed: %v", err)
	}

	gzWriter := gzip.NewWriter(file)
	if err := json.NewEncoder(gzWriter).Encode(records); err != nil {
		t.Fatalf("encode gzip json failed: %v", err)
	}
	if err := gzWriter.Close(); err != nil {
		t.Fatalf("close gzip writer failed: %v", err)
	}
	if err := file.Close(); err != nil {
		t.Fatalf("close file failed: %v", err)
	}

	dataset, err := LoadReferenceDatasetFromGzip(path)
	if err != nil {
		t.Fatalf("load dataset from gzip failed: %v", err)
	}

	if len(dataset.labels) != 2 {
		t.Fatalf("unexpected labels length: %d", len(dataset.labels))
	}
	if len(dataset.vectors) != 2*VectorDimensions {
		t.Fatalf("unexpected vectors length: %d", len(dataset.vectors))
	}
}

func vectorWithDim0(value float64) []float64 {
	vector := make([]float64, VectorDimensions)
	vector[0] = value
	return vector
}

func TestEuclideanDistanceSquared(t *testing.T) {
	query := vectorWithDim0(1)
	ref := make([]float32, VectorDimensions)
	ref[0] = 0

	got := euclideanDistanceSquared(query, ref)
	if math.Abs(got-1) > 1e-12 {
		t.Fatalf("unexpected distance: got %v want 1", got)
	}
}
