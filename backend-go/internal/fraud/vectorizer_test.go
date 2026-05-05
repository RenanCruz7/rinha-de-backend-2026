package fraud

import (
	"math"
	"testing"
)

func TestRealVectorizerLastTransactionNull(t *testing.T) {
	vectorizer := mustVectorizer(t)
	req := FraudScoreRequest{
		ID: "tx-1329056812",
		Transaction: Transaction{
			Amount:       41.12,
			Installments: 2,
			RequestedAt:  "2026-03-11T18:45:53Z",
		},
		Customer: Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: Merchant{
			ID:        "MERC-016",
			MCC:       "5411",
			AvgAmount: 60.25,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  29.23,
		},
		LastTransaction: nil,
	}

	got, err := vectorizer.Vectorize(req)
	if err != nil {
		t.Fatalf("vectorize failed: %v", err)
	}

	want := []float64{
		0.004112,
		0.16666666666666666,
		0.05,
		0.782608695652174,
		0.3333333333333333,
		-1,
		-1,
		0.02923,
		0.15,
		0,
		1,
		0,
		0.15,
		0.006025,
	}

	assertVectorAlmostEqual(t, got, want)
}

func TestRealVectorizerUsesDefaultMCCRiskWhenUnknown(t *testing.T) {
	vectorizer := mustVectorizer(t)
	req := baseRequest()
	req.Merchant.MCC = "0000"

	got, err := vectorizer.Vectorize(req)
	if err != nil {
		t.Fatalf("vectorize failed: %v", err)
	}

	if got[12] != DefaultMCCRisk {
		t.Fatalf("expected default mcc risk %v, got %v", DefaultMCCRisk, got[12])
	}
}

func TestRealVectorizerAppliesClamp(t *testing.T) {
	vectorizer := mustVectorizer(t)
	req := FraudScoreRequest{
		ID: "tx-clamp",
		Transaction: Transaction{
			Amount:       999999,
			Installments: 99,
			RequestedAt:  "2026-03-11T23:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      1,
			TxCount24h:     999,
			KnownMerchants: []string{},
		},
		Merchant: Merchant{
			ID:        "MERC-999",
			MCC:       "7802",
			AvgAmount: 999999,
		},
		Terminal: Terminal{
			IsOnline:    true,
			CardPresent: false,
			KmFromHome:  999999,
		},
		LastTransaction: &LastTransaction{
			Timestamp:     "2026-03-08T00:00:00Z",
			KmFromCurrent: 999999,
		},
	}

	got, err := vectorizer.Vectorize(req)
	if err != nil {
		t.Fatalf("vectorize failed: %v", err)
	}

	clampedIndices := []int{0, 1, 2, 5, 6, 7, 8, 13}
	for _, idx := range clampedIndices {
		if got[idx] != 1 {
			t.Fatalf("expected index %d to be clamped to 1, got %v", idx, got[idx])
		}
	}
}

func TestRealVectorizerUTCAndLastTransactionValues(t *testing.T) {
	vectorizer := mustVectorizer(t)
	req := FraudScoreRequest{
		ID: "tx-utc",
		Transaction: Transaction{
			Amount:       500,
			Installments: 6,
			RequestedAt:  "2026-03-10T03:00:00-03:00",
		},
		Customer: Customer{
			AvgAmount:      250,
			TxCount24h:     10,
			KnownMerchants: []string{"MERC-001"},
		},
		Merchant: Merchant{
			ID:        "MERC-002",
			MCC:       "5411",
			AvgAmount: 200,
		},
		Terminal: Terminal{
			IsOnline:    true,
			CardPresent: true,
			KmFromHome:  100,
		},
		LastTransaction: &LastTransaction{
			Timestamp:     "2026-03-10T02:30:00-03:00",
			KmFromCurrent: 50,
		},
	}

	got, err := vectorizer.Vectorize(req)
	if err != nil {
		t.Fatalf("vectorize failed: %v", err)
	}

	if got[3] != (6.0 / 23.0) {
		t.Fatalf("expected UTC hour index 3 to be 6/23, got %v", got[3])
	}
	if got[4] != (1.0 / 6.0) {
		t.Fatalf("expected Tuesday index 4 to be 1/6, got %v", got[4])
	}
	if got[5] != (30.0 / 1440.0) {
		t.Fatalf("expected minutes_since_last index 5 to be 30/1440, got %v", got[5])
	}
	if got[6] != (50.0 / 1000.0) {
		t.Fatalf("expected km_from_last_tx index 6 to be 50/1000, got %v", got[6])
	}
}

func mustVectorizer(t *testing.T) *RealVectorizer {
	t.Helper()

	params := NormalizationParams{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKM:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	risk := map[string]float64{
		"5411": 0.15,
		"7802": 0.75,
	}

	vectorizer, err := NewRealVectorizer(params, risk)
	if err != nil {
		t.Fatalf("failed to create vectorizer: %v", err)
	}
	return vectorizer
}

func baseRequest() FraudScoreRequest {
	return FraudScoreRequest{
		ID: "tx-base",
		Transaction: Transaction{
			Amount:       100,
			Installments: 1,
			RequestedAt:  "2026-03-11T00:00:00Z",
		},
		Customer: Customer{
			AvgAmount:      100,
			TxCount24h:     1,
			KnownMerchants: []string{"MERC-001"},
		},
		Merchant: Merchant{
			ID:        "MERC-001",
			MCC:       "5411",
			AvgAmount: 100,
		},
		Terminal: Terminal{
			IsOnline:    false,
			CardPresent: false,
			KmFromHome:  1,
		},
		LastTransaction: &LastTransaction{
			Timestamp:     "2026-03-10T23:00:00Z",
			KmFromCurrent: 1,
		},
	}
}

func assertVectorAlmostEqual(t *testing.T, got, want []float64) {
	t.Helper()

	if len(got) != len(want) {
		t.Fatalf("vector length mismatch: got %d want %d", len(got), len(want))
	}

	const epsilon = 1e-12
	for i := range want {
		if math.Abs(got[i]-want[i]) > epsilon {
			t.Fatalf("index %d mismatch: got %.15f want %.15f", i, got[i], want[i])
		}
	}
}
