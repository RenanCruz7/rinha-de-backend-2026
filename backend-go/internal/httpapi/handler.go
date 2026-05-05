package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/RenanCruz7/rinha-de-backend-2026/backend-go/internal/fraud"
)

var fallbackFraudScoreResponse = fraud.FraudScoreResponse{
	Approved:   true,
	FraudScore: 0.0,
}

type Handler struct {
	engine          fraud.Engine
	latencyRecorder *latencyRecorder
}

func NewHandler(engine fraud.Engine) *Handler {
	return &Handler{
		engine:          engine,
		latencyRecorder: newLatencyRecorder(defaultLatencyWindowSize),
	}
}

func (h *Handler) Routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", h.ready)
	mux.HandleFunc("POST /fraud-score", h.fraudScore)
	return mux
}

func (h *Handler) ready(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

func (h *Handler) fraudScore(w http.ResponseWriter, r *http.Request) {
	requestStart := time.Now()
	parseStart := requestStart

	var req fraud.FraudScoreRequest
	if err := decodeBody(r, &req); err != nil {
		h.recordAndLogLatency(latencySample{
			parse: time.Since(parseStart),
			total: time.Since(requestStart),
		}, "", "bad_request")
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid request payload",
		})
		return
	}

	if err := validateRequest(req); err != nil {
		h.recordAndLogLatency(latencySample{
			parse: time.Since(parseStart),
			total: time.Since(requestStart),
		}, req.ID, "bad_request")
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": err.Error(),
		})
		return
	}

	parseDuration := time.Since(parseStart)
	resp, evalTimings, err := h.evaluateSafely(r.Context(), req)
	usedFallback := false
	if err != nil {
		log.Printf("warning: fallback fraud-score response for id=%q: %v", req.ID, err)
		resp = fallbackFraudScoreResponse
		usedFallback = true
	}

	responseStart := time.Now()
	writeJSON(w, http.StatusOK, resp)
	responseDuration := time.Since(responseStart)
	totalDuration := time.Since(requestStart)

	h.recordAndLogLatency(latencySample{
		parse:     parseDuration,
		vectorize: evalTimings.Vectorize,
		search:    evalTimings.Search,
		decision:  evalTimings.Decision,
		response:  responseDuration,
		total:     totalDuration,
		fallback:  usedFallback,
	}, req.ID, "ok")
}

func (h *Handler) evaluateSafely(ctx context.Context, req fraud.FraudScoreRequest) (resp fraud.FraudScoreResponse, timings fraud.EvaluationTimings, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("panic during fraud evaluation: %v", recovered)
		}
	}()

	if timedEngine, ok := h.engine.(fraud.TimedEngine); ok {
		return timedEngine.EvaluateTimed(ctx, req)
	}

	resp, err = h.engine.Evaluate(ctx, req)
	return resp, timings, err
}

func (h *Handler) recordAndLogLatency(sample latencySample, requestID string, status string) {
	h.latencyRecorder.record(sample)
	log.Printf(
		"latency id=%q status=%s parse=%s vectorize=%s search=%s decision=%s response=%s total=%s fallback=%t",
		requestID,
		status,
		sample.parse,
		sample.vectorize,
		sample.search,
		sample.decision,
		sample.response,
		sample.total,
		sample.fallback,
	)

	snapshot := h.latencyRecorder.snapshot()
	if snapshot.requestCount > 0 && snapshot.requestCount%200 == 0 {
		log.Printf(
			"latency_p99 window=%d fallback_rate=%.4f total(p50=%s p95=%s p99=%s) parse(p99=%s) vectorize(p99=%s) search(p99=%s) decision(p99=%s) response(p99=%s)",
			defaultLatencyWindowSize,
			snapshot.fallbackRate,
			snapshot.total.p50,
			snapshot.total.p95,
			snapshot.total.p99,
			snapshot.parse.p99,
			snapshot.vectorize.p99,
			snapshot.search.p99,
			snapshot.decision.p99,
			snapshot.response.p99,
		)
	}
}

func decodeBody(r *http.Request, dst any) error {
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()

	if err := decoder.Decode(dst); err != nil {
		return err
	}

	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return errors.New("body must contain a single JSON object")
	}

	return nil
}

func validateRequest(req fraud.FraudScoreRequest) error {
	if strings.TrimSpace(req.ID) == "" {
		return errors.New("id is required")
	}
	if strings.TrimSpace(req.Transaction.RequestedAt) == "" {
		return errors.New("transaction.requested_at is required")
	}
	if strings.TrimSpace(req.Merchant.ID) == "" {
		return errors.New("merchant.id is required")
	}
	if strings.TrimSpace(req.Merchant.MCC) == "" {
		return errors.New("merchant.mcc is required")
	}
	return nil
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
