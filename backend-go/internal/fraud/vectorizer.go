package fraud

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
)

const (
	VectorDimensions = 14
	DefaultMCCRisk   = 0.5
)

type NormalizationParams struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKM                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

func (p NormalizationParams) Validate() error {
	if p.MaxAmount <= 0 {
		return errors.New("normalization.max_amount must be > 0")
	}
	if p.MaxInstallments <= 0 {
		return errors.New("normalization.max_installments must be > 0")
	}
	if p.AmountVsAvgRatio <= 0 {
		return errors.New("normalization.amount_vs_avg_ratio must be > 0")
	}
	if p.MaxMinutes <= 0 {
		return errors.New("normalization.max_minutes must be > 0")
	}
	if p.MaxKM <= 0 {
		return errors.New("normalization.max_km must be > 0")
	}
	if p.MaxTxCount24h <= 0 {
		return errors.New("normalization.max_tx_count_24h must be > 0")
	}
	if p.MaxMerchantAvgAmount <= 0 {
		return errors.New("normalization.max_merchant_avg_amount must be > 0")
	}
	return nil
}

type RealVectorizer struct {
	normalization NormalizationParams
	mccRisk       map[string]float64
}

func NewRealVectorizer(params NormalizationParams, mccRisk map[string]float64) (*RealVectorizer, error) {
	if err := params.Validate(); err != nil {
		return nil, err
	}

	clonedRisk := make(map[string]float64, len(mccRisk))
	for mcc, risk := range mccRisk {
		key := strings.TrimSpace(mcc)
		if key == "" {
			continue
		}
		clonedRisk[key] = clamp01(risk)
	}

	return &RealVectorizer{
		normalization: params,
		mccRisk:       clonedRisk,
	}, nil
}

func NewRealVectorizerFromFiles(normalizationPath, mccRiskPath string) (*RealVectorizer, error) {
	params, err := LoadNormalizationParams(normalizationPath)
	if err != nil {
		return nil, err
	}

	riskMap, err := LoadMCCRisk(mccRiskPath)
	if err != nil {
		return nil, err
	}

	return NewRealVectorizer(params, riskMap)
}

func LoadDefaultRealVectorizer() (*RealVectorizer, error) {
	type files struct {
		normalization string
		mccRisk       string
	}

	candidates := []files{
		{
			normalization: "resources/normalization.json",
			mccRisk:       "resources/mcc_risk.json",
		},
		{
			normalization: "../resources/normalization.json",
			mccRisk:       "../resources/mcc_risk.json",
		},
	}

	var lastErr error
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate.normalization); err != nil {
			lastErr = err
			continue
		}
		if _, err := os.Stat(candidate.mccRisk); err != nil {
			lastErr = err
			continue
		}

		vectorizer, err := NewRealVectorizerFromFiles(candidate.normalization, candidate.mccRisk)
		if err == nil {
			return vectorizer, nil
		}
		lastErr = err
	}

	return nil, fmt.Errorf("failed to load vectorization resources: %w", lastErr)
}

func LoadNormalizationParams(path string) (NormalizationParams, error) {
	var params NormalizationParams

	data, err := os.ReadFile(path)
	if err != nil {
		return NormalizationParams{}, fmt.Errorf("read normalization config: %w", err)
	}

	if err := json.Unmarshal(data, &params); err != nil {
		return NormalizationParams{}, fmt.Errorf("decode normalization config: %w", err)
	}

	return params, nil
}

func LoadMCCRisk(path string) (map[string]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read mcc risk config: %w", err)
	}

	var risk map[string]float64
	if err := json.Unmarshal(data, &risk); err != nil {
		return nil, fmt.Errorf("decode mcc risk config: %w", err)
	}

	return risk, nil
}

func (v *RealVectorizer) Vectorize(req FraudScoreRequest) ([]float64, error) {
	requestedAtUTC, err := parseRFC3339ToUTC(req.Transaction.RequestedAt)
	if err != nil {
		return nil, fmt.Errorf("invalid transaction.requested_at: %w", err)
	}

	vector := make([]float64, VectorDimensions)
	vector[0] = clamp01(req.Transaction.Amount / v.normalization.MaxAmount)
	vector[1] = clamp01(float64(req.Transaction.Installments) / v.normalization.MaxInstallments)
	vector[2] = clamp01((req.Transaction.Amount / req.Customer.AvgAmount) / v.normalization.AmountVsAvgRatio)
	vector[3] = float64(requestedAtUTC.Hour()) / 23.0
	vector[4] = float64(weekdayMondayZero(requestedAtUTC.Weekday())) / 6.0

	if req.LastTransaction == nil {
		vector[5] = -1
		vector[6] = -1
	} else {
		lastTxUTC, err := parseRFC3339ToUTC(req.LastTransaction.Timestamp)
		if err != nil {
			return nil, fmt.Errorf("invalid last_transaction.timestamp: %w", err)
		}

		minutesSinceLast := requestedAtUTC.Sub(lastTxUTC).Minutes()
		vector[5] = clamp01(minutesSinceLast / v.normalization.MaxMinutes)
		vector[6] = clamp01(req.LastTransaction.KmFromCurrent / v.normalization.MaxKM)
	}

	vector[7] = clamp01(req.Terminal.KmFromHome / v.normalization.MaxKM)
	vector[8] = clamp01(float64(req.Customer.TxCount24h) / v.normalization.MaxTxCount24h)
	vector[9] = boolToFloat(req.Terminal.IsOnline)
	vector[10] = boolToFloat(req.Terminal.CardPresent)
	vector[11] = unknownMerchantValue(req.Merchant.ID, req.Customer.KnownMerchants)
	vector[12] = v.lookupMCCRisk(req.Merchant.MCC)
	vector[13] = clamp01(req.Merchant.AvgAmount / v.normalization.MaxMerchantAvgAmount)

	return vector, nil
}

func (v *RealVectorizer) lookupMCCRisk(mcc string) float64 {
	risk, found := v.mccRisk[strings.TrimSpace(mcc)]
	if !found {
		return DefaultMCCRisk
	}
	return risk
}

func parseRFC3339ToUTC(raw string) (time.Time, error) {
	parsed, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return time.Time{}, err
	}
	return parsed.UTC(), nil
}

func weekdayMondayZero(weekday time.Weekday) int {
	return (int(weekday) + 6) % 7
}

func boolToFloat(value bool) float64 {
	if value {
		return 1
	}
	return 0
}

func unknownMerchantValue(merchantID string, knownMerchants []string) float64 {
	id := strings.TrimSpace(merchantID)
	for _, known := range knownMerchants {
		if id == strings.TrimSpace(known) {
			return 0
		}
	}
	return 1
}

func clamp01(value float64) float64 {
	if math.IsNaN(value) {
		return 0
	}
	if value < 0 {
		return 0
	}
	if value > 1 {
		return 1
	}
	return value
}
