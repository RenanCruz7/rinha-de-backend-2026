package httpapi

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/RenanCruz7/rinha-de-backend-2026/backend-go/internal/fraud"
)

type Handler struct {
	engine fraud.Engine
}

func NewHandler(engine fraud.Engine) *Handler {
	return &Handler{engine: engine}
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
	var req fraud.FraudScoreRequest
	if err := decodeBody(r, &req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": "invalid request payload",
		})
		return
	}

	if err := validateRequest(req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{
			"error": err.Error(),
		})
		return
	}

	resp, err := h.engine.Evaluate(r.Context(), req)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{
			"error": "failed to evaluate fraud score",
		})
		return
	}

	writeJSON(w, http.StatusOK, resp)
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
