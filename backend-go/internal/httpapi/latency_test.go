package httpapi

import (
	"testing"
	"time"
)

func TestLatencyRecorderSnapshotQuantiles(t *testing.T) {
	recorder := newLatencyRecorder(10)

	for i := 1; i <= 10; i++ {
		recorder.record(latencySample{
			parse:     time.Duration(i) * time.Millisecond,
			vectorize: time.Duration(i) * time.Millisecond,
			search:    time.Duration(i) * time.Millisecond,
			decision:  time.Duration(i) * time.Millisecond,
			response:  time.Duration(i) * time.Millisecond,
			total:     time.Duration(i) * time.Millisecond,
			fallback:  i%2 == 0,
		})
	}

	snapshot := recorder.snapshot()
	if snapshot.requestCount != 10 {
		t.Fatalf("unexpected request count: got %d want 10", snapshot.requestCount)
	}
	if snapshot.total.p50 != 5*time.Millisecond {
		t.Fatalf("unexpected p50: got %s want 5ms", snapshot.total.p50)
	}
	if snapshot.total.p95 != 10*time.Millisecond {
		t.Fatalf("unexpected p95: got %s want 10ms", snapshot.total.p95)
	}
	if snapshot.total.p99 != 10*time.Millisecond {
		t.Fatalf("unexpected p99: got %s want 10ms", snapshot.total.p99)
	}
	if snapshot.fallbackRate != 0.5 {
		t.Fatalf("unexpected fallback rate: got %.2f want 0.50", snapshot.fallbackRate)
	}
}

func TestDurationWindowRollingBehavior(t *testing.T) {
	window := newDurationWindow(3)

	window.add(1 * time.Millisecond)
	window.add(2 * time.Millisecond)
	window.add(3 * time.Millisecond)
	window.add(4 * time.Millisecond)

	quantiles := window.quantiles()
	if quantiles.p50 != 3*time.Millisecond {
		t.Fatalf("unexpected rolling p50: got %s want 3ms", quantiles.p50)
	}
	if quantiles.p99 != 4*time.Millisecond {
		t.Fatalf("unexpected rolling p99: got %s want 4ms", quantiles.p99)
	}
}
