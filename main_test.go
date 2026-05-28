package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

func TestSyncPIModelContext(t *testing.T) {
	dir := t.TempDir()
	modelsPath := filepath.Join(dir, "models.json")

	initial := map[string]any{
		"providers": map[string]any{
			"llm": map[string]any{
				"models": []any{
					map[string]any{"id": "qwen3.6-27b-local", "contextWindow": 1},
					map[string]any{"id": "qwen2.5-coder-7b-q5-local", "reasoning": true},
					map[string]any{"id": "other", "contextWindow": 123},
				},
			},
		},
	}
	b, _ := json.Marshal(initial)
	if err := os.WriteFile(modelsPath, b, 0o600); err != nil {
		t.Fatal(err)
	}

	routes := []RouteConfig{
		{ID: "qwen3.6-27b-local", ContextWindow: 65536},
		{ID: "qwen2.5-coder-7b-q5-local", ContextWindow: 65536},
	}
	if err := syncPIModelContext(modelsPath, routes); err != nil {
		t.Fatal(err)
	}

	out, err := os.ReadFile(modelsPath)
	if err != nil {
		t.Fatal(err)
	}
	var root map[string]any
	if err := json.Unmarshal(out, &root); err != nil {
		t.Fatal(err)
	}
	providers := root["providers"].(map[string]any)
	llm := providers["llm"].(map[string]any)
	models := llm["models"].([]any)

	got := map[string]int{}
	for _, m := range models {
		mm := m.(map[string]any)
		id := mm["id"].(string)
		if cw, ok := mm["contextWindow"].(float64); ok {
			got[id] = int(cw)
		}
	}

	if got["qwen3.6-27b-local"] != 65536 {
		t.Fatalf("qwen3.6-27b-local contextWindow mismatch: %d", got["qwen3.6-27b-local"])
	}
	if got["qwen2.5-coder-7b-q5-local"] != 65536 {
		t.Fatalf("qwen2.5-coder-7b-q5-local contextWindow mismatch: %d", got["qwen2.5-coder-7b-q5-local"])
	}
	if got["other"] != 123 {
		t.Fatalf("other model should be unchanged, got: %d", got["other"])
	}
}

func TestSyncPIModelContextAddsMissingRouteModel(t *testing.T) {
	dir := t.TempDir()
	modelsPath := filepath.Join(dir, "models.json")

	initial := map[string]any{
		"providers": map[string]any{
			"llm": map[string]any{
				"models": []any{
					map[string]any{"id": "qwen3.6-27b-local", "contextWindow": 1},
				},
			},
		},
	}
	b, _ := json.Marshal(initial)
	if err := os.WriteFile(modelsPath, b, 0o600); err != nil {
		t.Fatal(err)
	}

	routes := []RouteConfig{
		{ID: "qwen3.6-27b-local", ContextWindow: 65536},
		{ID: "qwen2.5-coder-7b-q5-local", ContextWindow: 131072},
	}
	if err := syncPIModelContext(modelsPath, routes); err != nil {
		t.Fatal(err)
	}

	out, err := os.ReadFile(modelsPath)
	if err != nil {
		t.Fatal(err)
	}
	var root map[string]any
	if err := json.Unmarshal(out, &root); err != nil {
		t.Fatal(err)
	}
	providers := root["providers"].(map[string]any)
	llm := providers["llm"].(map[string]any)
	models := llm["models"].([]any)

	var added map[string]any
	for _, m := range models {
		mm := m.(map[string]any)
		if mm["id"] == "qwen2.5-coder-7b-q5-local" {
			added = mm
			break
		}
	}
	if added == nil {
		t.Fatalf("expected missing route model to be added")
	}
	if got, ok := added["contextWindow"].(float64); !ok || int(got) != 131072 {
		t.Fatalf("unexpected contextWindow for added model: %#v", added["contextWindow"])
	}
	if got, ok := added["reasoning"].(bool); !ok || !got {
		t.Fatalf("expected reasoning=true for added model, got %#v", added["reasoning"])
	}
}

func TestExtractModelID(t *testing.T) {
	body := []byte(`{"model":"qwen3.6-27b-local"}`)
	if got := extractModelID("application/json", body); got != "qwen3.6-27b-local" {
		t.Fatalf("unexpected model id: %s", got)
	}
	if got := extractModelID("text/plain", body); got != "" {
		t.Fatalf("expected empty model id for non-json content-type")
	}
}

func TestHFDownloadURLPreservesRepoSlash(t *testing.T) {
	hf := HFConfig{
		Repo:     "Qwen/Qwen3.6-27B-GGUF",
		Revision: "main",
		Filename: "Qwen3.6-27B-Q4_K_M.gguf",
	}
	got := hfDownloadURL(hf)
	want := "https://huggingface.co/Qwen/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true"
	if got != want {
		t.Fatalf("unexpected hf url\nwant: %s\n got: %s", want, got)
	}
}

func TestDoUpstreamWithRetryOn503(t *testing.T) {
	var n int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := atomic.AddInt32(&n, 1)
		if call <= 2 {
			w.WriteHeader(http.StatusServiceUnavailable)
			_, _ = w.Write([]byte("Loading model"))
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	r := &Router{client: &http.Client{}}
	resp, err := r.doUpstreamWithRetry(t.Context(), srv.URL, http.MethodPost, "/v1/chat/completions", http.Header{
		"Content-Type": []string{"application/json"},
	}, []byte(`{"model":"qwen3.6-27b-local"}`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected 200, got %d", resp.StatusCode)
	}
	if got := atomic.LoadInt32(&n); got != 3 {
		t.Fatalf("expected 3 attempts, got %d", got)
	}
}
