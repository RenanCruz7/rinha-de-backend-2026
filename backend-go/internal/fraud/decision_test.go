package fraud

import "testing"

func TestThresholdDecisionPolicy_ZeroFraudOfFive(t *testing.T) {
	policy := NewThresholdDecisionPolicy()

	neighbors := []LabeledVector{
		{Label: "legit"},
		{Label: "legit"},
		{Label: "legit"},
		{Label: "legit"},
		{Label: "legit"},
	}

	got, err := policy.Decide(neighbors)
	if err != nil {
		t.Fatalf("decide failed: %v", err)
	}

	if !got.Approved || got.FraudScore != 0.0 {
		t.Fatalf("unexpected result: %+v", got)
	}
}

func TestThresholdDecisionPolicy_ThreeFraudOfFive(t *testing.T) {
	policy := NewThresholdDecisionPolicy()

	neighbors := []LabeledVector{
		{Label: "fraud"},
		{Label: "legit"},
		{Label: "fraud"},
		{Label: "legit"},
		{Label: "fraud"},
	}

	got, err := policy.Decide(neighbors)
	if err != nil {
		t.Fatalf("decide failed: %v", err)
	}

	if got.Approved || got.FraudScore != 0.6 {
		t.Fatalf("unexpected result: %+v", got)
	}
}

func TestThresholdDecisionPolicy_FiveFraudOfFive(t *testing.T) {
	policy := NewThresholdDecisionPolicy()

	neighbors := []LabeledVector{
		{Label: "fraud"},
		{Label: "fraud"},
		{Label: "fraud"},
		{Label: "fraud"},
		{Label: "fraud"},
	}

	got, err := policy.Decide(neighbors)
	if err != nil {
		t.Fatalf("decide failed: %v", err)
	}

	if got.Approved || got.FraudScore != 1.0 {
		t.Fatalf("unexpected result: %+v", got)
	}
}

func TestThresholdDecisionPolicy_RequiresExactlyFiveNeighbors(t *testing.T) {
	policy := NewThresholdDecisionPolicy()

	neighbors := []LabeledVector{
		{Label: "fraud"},
		{Label: "legit"},
		{Label: "fraud"},
		{Label: "legit"},
	}

	_, err := policy.Decide(neighbors)
	if err == nil {
		t.Fatal("expected error for non-5 neighbors, got nil")
	}
}
