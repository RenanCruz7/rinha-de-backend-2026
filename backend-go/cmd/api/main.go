package main

import (
	"log"
	"net/http"
	"time"

	"github.com/RenanCruz7/rinha-de-backend-2026/backend-go/internal/fraud"
	httpapi "github.com/RenanCruz7/rinha-de-backend-2026/backend-go/internal/httpapi"
)

func main() {
	engine := buildEngine()
	handler := httpapi.NewHandler(engine)

	server := &http.Server{
		Addr:              ":9999",
		Handler:           handler.Routes(),
		ReadHeaderTimeout: 2 * time.Second,
	}

	log.Printf("listening on %s", server.Addr)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}

func buildEngine() fraud.Engine {
	vectorizer, err := fraud.LoadDefaultRealVectorizer()
	if err != nil {
		log.Printf("warning: using stub engine because real vectorizer failed to load: %v", err)
		return fraud.NewStubEngine()
	}

	referenceDataset, err := fraud.LoadDefaultReferenceDataset()
	if err != nil {
		log.Printf("warning: using stub engine because references dataset failed to load: %v", err)
		return fraud.NewStubEngine()
	}

	searcher, err := fraud.NewBruteForceNeighborSearcher(referenceDataset)
	if err != nil {
		log.Printf("warning: using stub engine because searcher failed to build: %v", err)
		return fraud.NewStubEngine()
	}

	engine, err := fraud.NewPipelineEngine(
		vectorizer,
		searcher,
		fraud.NewThresholdDecisionPolicy(),
	)
	if err != nil {
		log.Printf("warning: using stub engine because pipeline failed to build: %v", err)
		return fraud.NewStubEngine()
	}

	return engine
}
