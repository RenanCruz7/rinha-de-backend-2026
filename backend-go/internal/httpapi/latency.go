package httpapi

import (
	"math"
	"sort"
	"sync"
	"time"
)

const defaultLatencyWindowSize = 1024

type latencySample struct {
	parse     time.Duration
	vectorize time.Duration
	search    time.Duration
	decision  time.Duration
	response  time.Duration
	total     time.Duration
	fallback  bool
}

type latencySnapshot struct {
	requestCount uint64
	fallbackRate float64
	parse        quantiles
	vectorize    quantiles
	search       quantiles
	decision     quantiles
	response     quantiles
	total        quantiles
}

type quantiles struct {
	p50 time.Duration
	p95 time.Duration
	p99 time.Duration
}

type latencyRecorder struct {
	mu            sync.Mutex
	parseSamples  durationWindow
	vecSamples    durationWindow
	searchSamples durationWindow
	decSamples    durationWindow
	respSamples   durationWindow
	totalSamples  durationWindow
	requestCount  uint64
	fallbackCount uint64
}

func newLatencyRecorder(windowSize int) *latencyRecorder {
	if windowSize <= 0 {
		windowSize = defaultLatencyWindowSize
	}

	return &latencyRecorder{
		parseSamples:  newDurationWindow(windowSize),
		vecSamples:    newDurationWindow(windowSize),
		searchSamples: newDurationWindow(windowSize),
		decSamples:    newDurationWindow(windowSize),
		respSamples:   newDurationWindow(windowSize),
		totalSamples:  newDurationWindow(windowSize),
	}
}

func (r *latencyRecorder) record(sample latencySample) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.requestCount++
	if sample.fallback {
		r.fallbackCount++
	}

	r.parseSamples.add(sample.parse)
	r.vecSamples.add(sample.vectorize)
	r.searchSamples.add(sample.search)
	r.decSamples.add(sample.decision)
	r.respSamples.add(sample.response)
	r.totalSamples.add(sample.total)
}

func (r *latencyRecorder) snapshot() latencySnapshot {
	r.mu.Lock()
	defer r.mu.Unlock()

	fallbackRate := 0.0
	if r.requestCount > 0 {
		fallbackRate = float64(r.fallbackCount) / float64(r.requestCount)
	}

	return latencySnapshot{
		requestCount: r.requestCount,
		fallbackRate: fallbackRate,
		parse:        r.parseSamples.quantiles(),
		vectorize:    r.vecSamples.quantiles(),
		search:       r.searchSamples.quantiles(),
		decision:     r.decSamples.quantiles(),
		response:     r.respSamples.quantiles(),
		total:        r.totalSamples.quantiles(),
	}
}

type durationWindow struct {
	values []time.Duration
	next   int
	filled bool
}

func newDurationWindow(size int) durationWindow {
	return durationWindow{values: make([]time.Duration, size)}
}

func (w *durationWindow) add(value time.Duration) {
	w.values[w.next] = value
	w.next++
	if w.next >= len(w.values) {
		w.next = 0
		w.filled = true
	}
}

func (w *durationWindow) currentValues() []time.Duration {
	if !w.filled {
		return append([]time.Duration(nil), w.values[:w.next]...)
	}
	return append([]time.Duration(nil), w.values...)
}

func (w *durationWindow) quantiles() quantiles {
	values := w.currentValues()
	if len(values) == 0 {
		return quantiles{}
	}

	sort.Slice(values, func(i, j int) bool {
		return values[i] < values[j]
	})

	return quantiles{
		p50: values[quantileIndex(len(values), 0.50)],
		p95: values[quantileIndex(len(values), 0.95)],
		p99: values[quantileIndex(len(values), 0.99)],
	}
}

func quantileIndex(size int, q float64) int {
	if size <= 1 {
		return 0
	}

	index := int(math.Ceil(float64(size)*q)) - 1
	if index < 0 {
		return 0
	}
	if index >= size {
		return size - 1
	}
	return index
}
