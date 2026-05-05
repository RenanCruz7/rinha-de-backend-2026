package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/RenanCruz7/rinha-de-backend-2026/backend-go/internal/fraud"
)

type engineStub struct {
	response fraud.FraudScoreResponse
	err      error
}

func (e engineStub) Evaluate(_ context.Context, _ fraud.FraudScoreRequest) (fraud.FraudScoreResponse, error) {
	return e.response, e.err
}

type panicEngineStub struct{}

func (panicEngineStub) Evaluate(_ context.Context, _ fraud.FraudScoreRequest) (fraud.FraudScoreResponse, error) {
	panic("unexpected panic from engine")
}

func TestReady(t *testing.T) {
	h := NewHandler(engineStub{})
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()

	h.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rr.Code)
	}
}

func TestFraudScoreValidPayload(t *testing.T) {
	h := NewHandler(engineStub{
		response: fraud.FraudScoreResponse{
			Approved:   true,
			FraudScore: 0,
		},
	})

	payload := `{
		"id":"tx-1",
		"transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},
		"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001"]},
		"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},
		"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},
		"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8}
	}`

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewBufferString(payload))
	rr := httptest.NewRecorder()

	h.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rr.Code)
	}

	var got fraud.FraudScoreResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if got.Approved != true || got.FraudScore != 0 {
		t.Fatalf("unexpected response: %+v", got)
	}
}

func TestFraudScoreInvalidPayload(t *testing.T) {
	h := NewHandler(engineStub{})
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewBufferString(`{"id":`))
	rr := httptest.NewRecorder()

	h.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", rr.Code)
	}
}

func TestFraudScoreMissingID(t *testing.T) {
	h := NewHandler(engineStub{})
	payload := `{
		"id":"",
		"transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},
		"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009"]},
		"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},
		"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},
		"last_transaction":null
	}`

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewBufferString(payload))
	rr := httptest.NewRecorder()
	h.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", rr.Code)
	}
}

func TestFraudScoreEngineError(t *testing.T) {
	h := NewHandler(engineStub{err: errors.New("boom")})
	payload := `{
		"id":"tx-1",
		"transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},
		"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009"]},
		"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},
		"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},
		"last_transaction":null
	}`

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewBufferString(payload))
	rr := httptest.NewRecorder()
	h.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rr.Code)
	}

	var got fraud.FraudScoreResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if got.Approved != true || got.FraudScore != 0 {
		t.Fatalf("unexpected fallback response: %+v", got)
	}
}

func TestFraudScoreEnginePanicFallback(t *testing.T) {
	h := NewHandler(panicEngineStub{})
	payload := `{
		"id":"tx-1",
		"transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},
		"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009"]},
		"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},
		"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7},
		"last_transaction":null
	}`

	req := httptest.NewRequest(http.MethodPost, "/fraud-score", bytes.NewBufferString(payload))
	rr := httptest.NewRecorder()
	h.Routes().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rr.Code)
	}

	var got fraud.FraudScoreResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &got); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if got.Approved != true || got.FraudScore != 0 {
		t.Fatalf("unexpected fallback response: %+v", got)
	}
}
