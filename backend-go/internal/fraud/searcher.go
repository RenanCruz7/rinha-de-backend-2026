package fraud

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

type BruteForceNeighborSearcher struct {
	dataset *ReferenceDataset
}

type ReferenceDataset struct {
	vectors []float32
	labels  []string
}

type referenceRecord struct {
	Vector []float64 `json:"vector"`
	Label  string    `json:"label"`
}

func NewBruteForceNeighborSearcher(dataset *ReferenceDataset) (*BruteForceNeighborSearcher, error) {
	if dataset == nil {
		return nil, errors.New("dataset is required")
	}
	if len(dataset.labels) == 0 {
		return nil, errors.New("dataset is empty")
	}
	expectedVectorLen := len(dataset.labels) * VectorDimensions
	if len(dataset.vectors) != expectedVectorLen {
		return nil, errors.New("dataset vector size mismatch")
	}
	return &BruteForceNeighborSearcher{dataset: dataset}, nil
}

func NewBruteForceNeighborSearcherFromGzip(path string) (*BruteForceNeighborSearcher, error) {
	dataset, err := LoadReferenceDatasetFromGzip(path)
	if err != nil {
		return nil, err
	}
	return NewBruteForceNeighborSearcher(dataset)
}

func LoadDefaultReferenceDataset() (*ReferenceDataset, error) {
	candidates := []string{
		"resources/references.json.gz",
		"../resources/references.json.gz",
	}

	var lastErr error
	for _, path := range candidates {
		if _, err := os.Stat(path); err != nil {
			lastErr = err
			continue
		}

		dataset, err := LoadReferenceDatasetFromGzip(path)
		if err == nil {
			return dataset, nil
		}
		lastErr = err
	}

	return nil, fmt.Errorf("failed to load references dataset: %w", lastErr)
}

func LoadReferenceDatasetFromGzip(path string) (*ReferenceDataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open references file: %w", err)
	}
	defer file.Close()

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("open gzip reader: %w", err)
	}
	defer gzReader.Close()

	decoder := json.NewDecoder(gzReader)

	token, err := decoder.Token()
	if err != nil {
		return nil, fmt.Errorf("read json start token: %w", err)
	}

	delimiter, ok := token.(json.Delim)
	if !ok || delimiter != '[' {
		return nil, errors.New("references root must be an array")
	}

	records := make([]referenceRecord, 0, 1024)
	for decoder.More() {
		var record referenceRecord
		if err := decoder.Decode(&record); err != nil {
			return nil, fmt.Errorf("decode reference record: %w", err)
		}
		records = append(records, record)
	}

	if _, err := decoder.Token(); err != nil {
		return nil, fmt.Errorf("read json end token: %w", err)
	}

	return buildReferenceDataset(records)
}

func buildReferenceDataset(records []referenceRecord) (*ReferenceDataset, error) {
	if len(records) == 0 {
		return nil, errors.New("reference dataset is empty")
	}

	vectors := make([]float32, 0, len(records)*VectorDimensions)
	labels := make([]string, 0, len(records))

	for i, record := range records {
		if len(record.Vector) != VectorDimensions {
			return nil, fmt.Errorf("reference record %d has invalid vector length: got %d want %d", i, len(record.Vector), VectorDimensions)
		}

		label := strings.TrimSpace(record.Label)
		if label == "" {
			return nil, fmt.Errorf("reference record %d has empty label", i)
		}

		for _, value := range record.Vector {
			vectors = append(vectors, float32(value))
		}
		labels = append(labels, label)
	}

	return &ReferenceDataset{
		vectors: vectors,
		labels:  labels,
	}, nil
}

func (s *BruteForceNeighborSearcher) FindKNearest(ctx context.Context, vector []float64, k int) ([]LabeledVector, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(vector) != VectorDimensions {
		return nil, fmt.Errorf("query vector must have %d dimensions", VectorDimensions)
	}
	if k <= 0 {
		return nil, errors.New("k must be > 0")
	}

	total := len(s.dataset.labels)
	if total == 0 {
		return nil, errors.New("dataset is empty")
	}

	if k > total {
		k = total
	}

	candidates := make([]neighborCandidate, 0, k)
	for idx := 0; idx < total; idx++ {
		if idx%10000 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}

		start := idx * VectorDimensions
		dist := euclideanDistanceSquared(vector, s.dataset.vectors[start:start+VectorDimensions])
		candidate := neighborCandidate{
			index:    idx,
			distance: dist,
		}

		if len(candidates) < k {
			candidates = append(candidates, candidate)
			if len(candidates) == k {
				sort.Slice(candidates, func(i, j int) bool {
					if candidates[i].distance != candidates[j].distance {
						return candidates[i].distance > candidates[j].distance
					}
					return candidates[i].index > candidates[j].index
				})
			}
			continue
		}

		if isCandidateBetter(candidate, candidates[0]) {
			candidates[0] = candidate
			sort.Slice(candidates, func(i, j int) bool {
				if candidates[i].distance != candidates[j].distance {
					return candidates[i].distance > candidates[j].distance
				}
				return candidates[i].index > candidates[j].index
			})
		}
	}

	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].distance != candidates[j].distance {
			return candidates[i].distance < candidates[j].distance
		}
		return candidates[i].index < candidates[j].index
	})

	neighbors := make([]LabeledVector, 0, len(candidates))
	for _, candidate := range candidates {
		neighbors = append(neighbors, LabeledVector{
			Label:    s.dataset.labels[candidate.index],
			Distance: math.Sqrt(candidate.distance),
		})
	}

	return neighbors, nil
}

type neighborCandidate struct {
	index    int
	distance float64
}

func euclideanDistanceSquared(query []float64, reference []float32) float64 {
	var sum float64
	for i := 0; i < VectorDimensions; i++ {
		diff := query[i] - float64(reference[i])
		sum += diff * diff
	}
	return sum
}

func isCandidateBetter(candidate neighborCandidate, currentWorst neighborCandidate) bool {
	if candidate.distance < currentWorst.distance {
		return true
	}
	if candidate.distance > currentWorst.distance {
		return false
	}
	return candidate.index < currentWorst.index
}
