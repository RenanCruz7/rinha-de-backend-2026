package fraud

import (
	"compress/gzip"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
)

const (
	defaultSearchWorkers     = 0 // 0 => auto (GOMAXPROCS)
	defaultCancelCheckStride = 1024
)

type BruteForceNeighborSearcher struct {
	dataset           *ReferenceDataset
	workers           int
	cancelCheckStride int
}

type ReferenceDataset struct {
	vectors    []float32
	labels     []string
	isFraudTag []bool
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
	if len(dataset.isFraudTag) != len(dataset.labels) {
		return nil, errors.New("dataset fraud tag size mismatch")
	}

	workers := resolveSearchWorkers()
	return &BruteForceNeighborSearcher{
		dataset:           dataset,
		workers:           workers,
		cancelCheckStride: defaultCancelCheckStride,
	}, nil
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
	isFraudTag := make([]bool, 0, len(records))

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
		isFraudTag = append(isFraudTag, strings.EqualFold(label, "fraud"))
	}

	return &ReferenceDataset{
		vectors:    vectors,
		labels:     labels,
		isFraudTag: isFraudTag,
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

	candidates, err := s.findKNearestCandidates(ctx, vector, k)
	if err != nil {
		return nil, err
	}

	neighbors := make([]LabeledVector, 0, len(candidates))
	for _, candidate := range candidates {
		neighbors = append(neighbors, LabeledVector{
			Label:    s.dataset.labels[candidate.index],
			IsFraud:  s.dataset.isFraudTag[candidate.index],
			Distance: math.Sqrt(candidate.distance),
		})
	}

	return neighbors, nil
}

func (s *BruteForceNeighborSearcher) findKNearestCandidates(ctx context.Context, query []float64, k int) ([]neighborCandidate, error) {
	total := len(s.dataset.labels)
	if total == 0 {
		return nil, errors.New("dataset is empty")
	}

	workers := s.workers
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers < 1 {
		workers = 1
	}
	if workers > total {
		workers = total
	}

	chunkSize := (total + workers - 1) / workers
	results := make(chan workerTopKResult, workers)

	var wg sync.WaitGroup
	for workerID := 0; workerID < workers; workerID++ {
		start := workerID * chunkSize
		if start >= total {
			break
		}
		end := start + chunkSize
		if end > total {
			end = total
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			results <- s.scanRangeTopK(ctx, query, k, start, end)
		}(start, end)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	globalTopK := newTopK(k)
	for result := range results {
		if result.err != nil {
			return nil, result.err
		}

		for _, candidate := range result.topK.candidates() {
			globalTopK.tryInsert(candidate)
		}
	}

	candidates := globalTopK.candidates()
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].distance != candidates[j].distance {
			return candidates[i].distance < candidates[j].distance
		}
		return candidates[i].index < candidates[j].index
	})

	return candidates, nil
}

func (s *BruteForceNeighborSearcher) scanRangeTopK(ctx context.Context, query []float64, k, start, end int) workerTopKResult {
	topK := newTopK(k)
	stride := s.cancelCheckStride
	if stride <= 0 {
		stride = defaultCancelCheckStride
	}

	for idx := start; idx < end; idx++ {
		if (idx-start)%stride == 0 {
			if err := ctx.Err(); err != nil {
				return workerTopKResult{err: err}
			}
		}

		vectorStart := idx * VectorDimensions
		reference := s.dataset.vectors[vectorStart : vectorStart+VectorDimensions]
		cutoff := topK.currentWorstDistance()
		distance := euclideanDistanceSquaredWithCutoff(query, reference, cutoff)

		topK.tryInsert(neighborCandidate{
			index:    idx,
			distance: distance,
		})
	}

	return workerTopKResult{topK: topK}
}

type neighborCandidate struct {
	index    int
	distance float64
}

type workerTopKResult struct {
	topK topKCandidates
	err  error
}

type topKCandidates struct {
	k     int
	items []neighborCandidate
	count int
}

func newTopK(k int) topKCandidates {
	return topKCandidates{
		k:     k,
		items: make([]neighborCandidate, k),
	}
}

func (t *topKCandidates) tryInsert(candidate neighborCandidate) {
	if t.k <= 0 {
		return
	}

	if t.count < t.k {
		t.items[t.count] = candidate
		t.count++
		return
	}

	worst := t.worstIndex()
	if isCandidateBetter(candidate, t.items[worst]) {
		t.items[worst] = candidate
	}
}

func (t *topKCandidates) currentWorstDistance() float64 {
	if t.count < t.k {
		return math.Inf(1)
	}
	return t.items[t.worstIndex()].distance
}

func (t *topKCandidates) worstIndex() int {
	worst := 0
	for i := 1; i < t.count; i++ {
		if isCandidateWorse(t.items[i], t.items[worst]) {
			worst = i
		}
	}
	return worst
}

func (t *topKCandidates) candidates() []neighborCandidate {
	return append([]neighborCandidate(nil), t.items[:t.count]...)
}

func euclideanDistanceSquared(query []float64, reference []float32) float64 {
	return euclideanDistanceSquaredWithCutoff(query, reference, math.Inf(1))
}

func euclideanDistanceSquaredWithCutoff(query []float64, reference []float32, cutoff float64) float64 {
	var sum float64
	for i := 0; i < VectorDimensions; i++ {
		diff := query[i] - float64(reference[i])
		sum += diff * diff
		if sum > cutoff {
			return sum
		}
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

func isCandidateWorse(candidate neighborCandidate, currentWorst neighborCandidate) bool {
	if candidate.distance > currentWorst.distance {
		return true
	}
	if candidate.distance < currentWorst.distance {
		return false
	}
	return candidate.index > currentWorst.index
}

func resolveSearchWorkers() int {
	raw := strings.TrimSpace(os.Getenv("SEARCH_WORKERS"))
	if raw == "" {
		return defaultSearchWorkers
	}

	workers, err := strconv.Atoi(raw)
	if err != nil || workers < 0 {
		return defaultSearchWorkers
	}
	return workers
}
