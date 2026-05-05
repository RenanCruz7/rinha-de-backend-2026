package fraud

import (
	"context"
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
	SearchModeBrute = "brute"
	SearchModeANN   = "ann"
)

const (
	defaultIVFCentroids     = 32
	defaultIVFNProbe        = 4
	defaultIVFMinCandidates = 10000
	defaultIVFSampleSize    = 120000
)

type IVFConfig struct {
	Centroids     int
	NProbe        int
	MinCandidates int
	SampleSize    int
}

type IVFNeighborSearcher struct {
	dataset           *ReferenceDataset
	fallback          *BruteForceNeighborSearcher
	centroids         []float32
	lists             [][]int
	nprobe            int
	minCandidates     int
	cancelCheckStride int
}

func NewNeighborSearcherFromEnv(dataset *ReferenceDataset) (NeighborSearcher, error) {
	mode := strings.ToLower(strings.TrimSpace(os.Getenv("SEARCH_MODE")))
	if mode == "" {
		mode = SearchModeBrute
	}

	switch mode {
	case SearchModeBrute:
		return NewBruteForceNeighborSearcher(dataset)
	case SearchModeANN:
		cfg := loadIVFConfigFromEnv()
		return NewIVFNeighborSearcher(dataset, cfg)
	default:
		return nil, fmt.Errorf("unknown SEARCH_MODE %q", mode)
	}
}

func NewIVFNeighborSearcher(dataset *ReferenceDataset, cfg IVFConfig) (*IVFNeighborSearcher, error) {
	if dataset == nil {
		return nil, errors.New("dataset is required")
	}
	if len(dataset.labels) == 0 {
		return nil, errors.New("dataset is empty")
	}

	total := len(dataset.labels)
	cfg = normalizeIVFConfig(cfg, total)

	fallback, err := NewBruteForceNeighborSearcher(dataset)
	if err != nil {
		return nil, err
	}

	centroids, err := selectInitialCentroids(dataset, cfg.Centroids)
	if err != nil {
		return nil, err
	}

	refineCentroidsFromSample(dataset, centroids, cfg.SampleSize)

	lists, err := assignVectorsToCentroids(dataset, centroids)
	if err != nil {
		return nil, err
	}

	return &IVFNeighborSearcher{
		dataset:           dataset,
		fallback:          fallback,
		centroids:         centroids,
		lists:             lists,
		nprobe:            cfg.NProbe,
		minCandidates:     cfg.MinCandidates,
		cancelCheckStride: defaultCancelCheckStride,
	}, nil
}

func (s *IVFNeighborSearcher) FindKNearest(ctx context.Context, query []float64, k int) ([]LabeledVector, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query) != VectorDimensions {
		return nil, fmt.Errorf("query vector must have %d dimensions", VectorDimensions)
	}
	if k <= 0 {
		return nil, errors.New("k must be > 0")
	}

	centroidCandidates := s.nearestCentroids(query, s.nprobe)
	candidateCount := 0
	for _, c := range centroidCandidates {
		candidateCount += len(s.lists[c.index])
	}

	// Preserve quality when the ANN shortlist is too small.
	if candidateCount < s.minCandidates {
		return s.fallback.FindKNearest(ctx, query, k)
	}

	topK := newTopK(k)
	stride := s.cancelCheckStride
	if stride <= 0 {
		stride = defaultCancelCheckStride
	}

	seen := 0
	for _, centroid := range centroidCandidates {
		list := s.lists[centroid.index]
		for _, vecIdx := range list {
			if seen%stride == 0 {
				if err := ctx.Err(); err != nil {
					return nil, err
				}
			}
			seen++

			start := vecIdx * VectorDimensions
			ref := s.dataset.vectors[start : start+VectorDimensions]
			distance := euclideanDistanceSquaredWithCutoff(query, ref, topK.currentWorstDistance())
			topK.tryInsert(neighborCandidate{
				index:    vecIdx,
				distance: distance,
			})
		}
	}

	candidates := topK.candidates()
	if len(candidates) < k {
		return s.fallback.FindKNearest(ctx, query, k)
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
			IsFraud:  s.dataset.isFraudTag[candidate.index],
			Distance: math.Sqrt(candidate.distance),
		})
	}

	return neighbors, nil
}

func (s *IVFNeighborSearcher) nearestCentroids(query []float64, nprobe int) []neighborCandidate {
	if nprobe <= 0 {
		nprobe = 1
	}
	totalCentroids := len(s.centroids) / VectorDimensions
	if nprobe > totalCentroids {
		nprobe = totalCentroids
	}

	top := newTopK(nprobe)
	for idx := 0; idx < totalCentroids; idx++ {
		start := idx * VectorDimensions
		centroid := s.centroids[start : start+VectorDimensions]
		distance := centroidDistanceSquared(query, centroid)
		top.tryInsert(neighborCandidate{
			index:    idx,
			distance: distance,
		})
	}

	candidates := top.candidates()
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].distance != candidates[j].distance {
			return candidates[i].distance < candidates[j].distance
		}
		return candidates[i].index < candidates[j].index
	})
	return candidates
}

func normalizeIVFConfig(cfg IVFConfig, total int) IVFConfig {
	if cfg.Centroids <= 0 {
		cfg.Centroids = defaultIVFCentroids
	}
	if cfg.Centroids > total {
		cfg.Centroids = total
	}

	if cfg.NProbe <= 0 {
		cfg.NProbe = defaultIVFNProbe
	}
	if cfg.NProbe > cfg.Centroids {
		cfg.NProbe = cfg.Centroids
	}

	if cfg.MinCandidates <= 0 {
		cfg.MinCandidates = defaultIVFMinCandidates
	}
	if cfg.MinCandidates > total {
		cfg.MinCandidates = total
	}

	if cfg.SampleSize <= 0 {
		cfg.SampleSize = defaultIVFSampleSize
	}
	if cfg.SampleSize > total {
		cfg.SampleSize = total
	}

	return cfg
}

func loadIVFConfigFromEnv() IVFConfig {
	return IVFConfig{
		Centroids:     envInt("IVF_CENTROIDS", defaultIVFCentroids),
		NProbe:        envInt("IVF_NPROBE", defaultIVFNProbe),
		MinCandidates: envInt("IVF_MIN_CANDIDATES", defaultIVFMinCandidates),
		SampleSize:    envInt("IVF_SAMPLE_SIZE", defaultIVFSampleSize),
	}
}

func envInt(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func selectInitialCentroids(dataset *ReferenceDataset, centroidCount int) ([]float32, error) {
	total := len(dataset.labels)
	if total == 0 {
		return nil, errors.New("dataset is empty")
	}
	if centroidCount <= 0 {
		return nil, errors.New("centroid count must be > 0")
	}
	if centroidCount > total {
		centroidCount = total
	}

	centroids := make([]float32, centroidCount*VectorDimensions)
	step := float64(total) / float64(centroidCount)
	for i := 0; i < centroidCount; i++ {
		idx := int(float64(i)*step + step/2)
		if idx >= total {
			idx = total - 1
		}
		srcStart := idx * VectorDimensions
		dstStart := i * VectorDimensions
		copy(centroids[dstStart:dstStart+VectorDimensions], dataset.vectors[srcStart:srcStart+VectorDimensions])
	}

	return centroids, nil
}

func refineCentroidsFromSample(dataset *ReferenceDataset, centroids []float32, sampleSize int) {
	totalCentroids := len(centroids) / VectorDimensions
	if totalCentroids == 0 || sampleSize <= 0 {
		return
	}

	sums := make([]float64, len(centroids))
	counts := make([]int, totalCentroids)

	totalVectors := len(dataset.labels)
	step := 1
	if sampleSize < totalVectors {
		step = totalVectors / sampleSize
		if step < 1 {
			step = 1
		}
	}

	sampled := 0
	for idx := 0; idx < totalVectors && sampled < sampleSize; idx += step {
		vecStart := idx * VectorDimensions
		vec := dataset.vectors[vecStart : vecStart+VectorDimensions]

		bestCentroid := 0
		bestDistance := math.Inf(1)
		for centroidID := 0; centroidID < totalCentroids; centroidID++ {
			centroidStart := centroidID * VectorDimensions
			dist := distanceFloat32ToFloat32(vec, centroids[centroidStart:centroidStart+VectorDimensions], bestDistance)
			if dist < bestDistance {
				bestDistance = dist
				bestCentroid = centroidID
			}
		}

		base := bestCentroid * VectorDimensions
		for d := 0; d < VectorDimensions; d++ {
			sums[base+d] += float64(vec[d])
		}
		counts[bestCentroid]++
		sampled++
	}

	for centroidID := 0; centroidID < totalCentroids; centroidID++ {
		count := counts[centroidID]
		if count == 0 {
			continue
		}
		base := centroidID * VectorDimensions
		for d := 0; d < VectorDimensions; d++ {
			centroids[base+d] = float32(sums[base+d] / float64(count))
		}
	}
}

func assignVectorsToCentroids(dataset *ReferenceDataset, centroids []float32) ([][]int, error) {
	totalVectors := len(dataset.labels)
	totalCentroids := len(centroids) / VectorDimensions
	if totalCentroids == 0 {
		return nil, errors.New("no centroids")
	}

	workers := runtime.GOMAXPROCS(0)
	if workers < 1 {
		workers = 1
	}
	if workers > totalVectors {
		workers = totalVectors
	}

	chunkSize := (totalVectors + workers - 1) / workers
	type assignmentResult struct {
		buckets [][]int
	}
	results := make(chan assignmentResult, workers)

	var wg sync.WaitGroup
	for workerID := 0; workerID < workers; workerID++ {
		start := workerID * chunkSize
		if start >= totalVectors {
			break
		}
		end := start + chunkSize
		if end > totalVectors {
			end = totalVectors
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			local := make([][]int, totalCentroids)
			for idx := start; idx < end; idx++ {
				vecStart := idx * VectorDimensions
				vec := dataset.vectors[vecStart : vecStart+VectorDimensions]

				bestCentroid := 0
				bestDistance := math.Inf(1)
				for centroidID := 0; centroidID < totalCentroids; centroidID++ {
					centroidStart := centroidID * VectorDimensions
					dist := distanceFloat32ToFloat32(vec, centroids[centroidStart:centroidStart+VectorDimensions], bestDistance)
					if dist < bestDistance {
						bestDistance = dist
						bestCentroid = centroidID
					}
				}

				local[bestCentroid] = append(local[bestCentroid], idx)
			}
			results <- assignmentResult{buckets: local}
		}(start, end)
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	lists := make([][]int, totalCentroids)
	for result := range results {
		for centroidID, indices := range result.buckets {
			if len(indices) == 0 {
				continue
			}
			lists[centroidID] = append(lists[centroidID], indices...)
		}
	}

	return lists, nil
}

func centroidDistanceSquared(query []float64, centroid []float32) float64 {
	var sum float64
	for i := 0; i < VectorDimensions; i++ {
		diff := query[i] - float64(centroid[i])
		sum += diff * diff
	}
	return sum
}

func distanceFloat32ToFloat32(a []float32, b []float32, cutoff float64) float64 {
	var sum float64
	for i := 0; i < VectorDimensions; i++ {
		diff := float64(a[i] - b[i])
		sum += diff * diff
		if sum > cutoff {
			return sum
		}
	}
	return sum
}
