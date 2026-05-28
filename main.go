package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

type Config struct {
	ListenAddr       string        `yaml:"listen_addr"`
	EnablePISync     bool          `yaml:"enable_pi_model_sync"`
	PIModelsJSONPath string        `yaml:"pi_models_json_path"`
	Routes           []RouteConfig `yaml:"routes"`
}

type RouteConfig struct {
	ID            string    `yaml:"id"`
	BackendURL    string    `yaml:"backend_url"`
	ContextWindow int       `yaml:"context_window"`
	ModelFilePath string    `yaml:"model_file_path"`
	HuggingFace   *HFConfig `yaml:"huggingface"`
}

type HFConfig struct {
	Repo       string `yaml:"repo"`
	Filename   string `yaml:"filename"`
	Revision   string `yaml:"revision"`
	TokenEnv   string `yaml:"token_env"`
	TargetPath string `yaml:"target_path"`
}

type Router struct {
	cfg        Config
	routesByID map[string]RouteConfig
	client     *http.Client
	dlMu       sync.Mutex
	dlLocks    map[string]*sync.Mutex
}

const (
	upstream503MaxRetries = 3
)

func main() {
	cfgPath := os.Getenv("LLM_ROUTER_CONFIG")
	if cfgPath == "" {
		cfgPath = "router.yaml"
	}

	cfg, err := loadConfig(cfgPath)
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	if cfg.EnablePISync {
		if err := syncPIModelContext(cfg.PIModelsJSONPath, cfg.Routes); err != nil {
			log.Printf("warning: pi model sync skipped: %v", err)
		}
	}

	r := &Router{
		cfg:        cfg,
		routesByID: make(map[string]RouteConfig, len(cfg.Routes)),
		client:     &http.Client{Timeout: 0},
		dlLocks:    map[string]*sync.Mutex{},
	}
	for _, rt := range cfg.Routes {
		r.routesByID[rt.ID] = rt
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", r.handleHealth)
	mux.HandleFunc("/v1/models", r.handleModels)
	mux.HandleFunc("/models", r.handleModels)
	mux.HandleFunc("/", r.handleProxy)

	log.Printf("llm-router listening on %s", cfg.ListenAddr)
	if err := http.ListenAndServe(cfg.ListenAddr, mux); err != nil {
		log.Fatal(err)
	}
}

func loadConfig(path string) (Config, error) {
	var cfg Config
	b, err := os.ReadFile(path)
	if err != nil {
		return cfg, err
	}
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return cfg, err
	}

	if cfg.ListenAddr == "" {
		p := os.Getenv("LLM_ROUTER_PORT")
		if p == "" {
			p = "8090"
		}
		cfg.ListenAddr = "127.0.0.1:" + p
	}
	if cfg.EnablePISync && cfg.PIModelsJSONPath == "" {
		cfg.PIModelsJSONPath = "~/.pi/agent/models.json"
	}
	if cfg.PIModelsJSONPath != "" {
		p, err := expandPath(cfg.PIModelsJSONPath)
		if err != nil {
			return cfg, err
		}
		cfg.PIModelsJSONPath = p
	}

	if len(cfg.Routes) == 0 {
		return cfg, errors.New("at least one route is required")
	}

	seen := map[string]bool{}
	for i, rt := range cfg.Routes {
		if rt.ID == "" || rt.BackendURL == "" {
			return cfg, fmt.Errorf("routes[%d]: id and backend_url are required", i)
		}
		if seen[rt.ID] {
			return cfg, fmt.Errorf("duplicate route id: %s", rt.ID)
		}
		seen[rt.ID] = true
		if _, err := url.ParseRequestURI(rt.BackendURL); err != nil {
			return cfg, fmt.Errorf("invalid backend_url for %s: %w", rt.ID, err)
		}
		if rt.ModelFilePath != "" {
			x, err := expandPath(rt.ModelFilePath)
			if err != nil {
				return cfg, err
			}
			rt.ModelFilePath = x
		}
		if rt.HuggingFace != nil {
			if rt.HuggingFace.Repo == "" || rt.HuggingFace.Filename == "" {
				return cfg, fmt.Errorf("route %s: huggingface.repo and huggingface.filename are required", rt.ID)
			}
			if rt.HuggingFace.Revision == "" {
				rt.HuggingFace.Revision = "main"
			}
			if rt.HuggingFace.TokenEnv == "" {
				rt.HuggingFace.TokenEnv = "HF_TOKEN"
			}
			if rt.HuggingFace.TargetPath == "" {
				rt.HuggingFace.TargetPath = rt.ModelFilePath
			}
			t, err := expandPath(rt.HuggingFace.TargetPath)
			if err != nil {
				return cfg, err
			}
			rt.HuggingFace.TargetPath = t
		}
		cfg.Routes[i] = rt
	}

	return cfg, nil
}

func expandPath(p string) (string, error) {
	if p == "" {
		return p, nil
	}
	if strings.HasPrefix(p, "~/") {
		h, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		return filepath.Join(h, strings.TrimPrefix(p, "~/")), nil
	}
	if p == "~" {
		return os.UserHomeDir()
	}
	return p, nil
}

func syncPIModelContext(path string, routes []RouteConfig) error {
	lockPath := path + ".lock"
	lockFile, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return err
	}
	defer lockFile.Close()
	if err := syscall.Flock(int(lockFile.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	defer syscall.Flock(int(lockFile.Fd()), syscall.LOCK_UN)

	b, err := os.ReadFile(path)
	if err != nil {
		return err
	}

	var root map[string]any
	if err := json.Unmarshal(b, &root); err != nil {
		return err
	}

	providers, ok := root["providers"].(map[string]any)
	if !ok {
		return errors.New("models.json missing providers object")
	}

	ctxByID := map[string]int{}
	for _, rt := range routes {
		if rt.ContextWindow > 0 {
			ctxByID[rt.ID] = rt.ContextWindow
		}
	}

	for _, pv := range providers {
		pmap, ok := pv.(map[string]any)
		if !ok {
			continue
		}
		models, ok := pmap["models"].([]any)
		if !ok {
			continue
		}
		for _, m := range models {
			mm, ok := m.(map[string]any)
			if !ok {
				continue
			}
			id, _ := mm["id"].(string)
			if cw, ok := ctxByID[id]; ok {
				mm["contextWindow"] = cw
			}
		}
	}

	// Ensure llm provider contains all routed model IDs (not only context updates).
	if llmProvider, ok := providers["llm"].(map[string]any); ok {
		if models, ok := llmProvider["models"].([]any); ok {
			seen := make(map[string]struct{}, len(models))
			for _, m := range models {
				mm, ok := m.(map[string]any)
				if !ok {
					continue
				}
				if id, ok := mm["id"].(string); ok && id != "" {
					seen[id] = struct{}{}
				}
			}
			for _, rt := range routes {
				if _, exists := seen[rt.ID]; exists {
					continue
				}
				entry := map[string]any{
					"id":        rt.ID,
					"input":     []any{"text"},
					"reasoning": true,
				}
				if rt.ContextWindow > 0 {
					entry["contextWindow"] = rt.ContextWindow
				}
				models = append(models, entry)
			}
			llmProvider["models"] = models
		}
	}

	out, err := json.MarshalIndent(root, "", "  ")
	if err != nil {
		return err
	}
	out = append(out, '\n')

	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, out, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (r *Router) handleHealth(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		sendJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method_not_allowed"})
		return
	}
	routes := map[string]string{}
	for _, rt := range r.cfg.Routes {
		routes[rt.ID] = rt.BackendURL
	}
	sendJSON(w, http.StatusOK, map[string]any{"ok": true, "routes": routes})
}

func (r *Router) handleModels(w http.ResponseWriter, req *http.Request) {
	if req.Method != http.MethodGet {
		sendJSON(w, http.StatusMethodNotAllowed, map[string]any{"error": "method_not_allowed"})
		return
	}
	data := make([]map[string]any, 0, len(r.cfg.Routes))
	for _, rt := range r.cfg.Routes {
		data = append(data, map[string]any{"id": rt.ID, "object": "model"})
	}
	sendJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

func (r *Router) handleProxy(w http.ResponseWriter, req *http.Request) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		sendRouterErr(w, http.StatusBadRequest, "invalid_body", err)
		return
	}
	_ = req.Body.Close()

	modelID := extractModelID(req.Header.Get("Content-Type"), body)
	rt, ok := r.routesByID[modelID]
	if !ok {
		sendJSON(w, http.StatusNotFound, map[string]any{
			"error": map[string]any{
				"type":    "not_found_error",
				"message": fmt.Sprintf("Model not routed: %s", modelID),
			},
		})
		return
	}

	if err := r.ensureModelAvailable(req.Context(), rt); err != nil {
		sendRouterErr(w, http.StatusServiceUnavailable, "download_failed", err)
		return
	}

	upstreamURL := strings.TrimRight(rt.BackendURL, "/") + req.URL.RequestURI()
	uReq, err := http.NewRequestWithContext(req.Context(), req.Method, upstreamURL, bytes.NewReader(body))
	if err != nil {
		sendRouterErr(w, http.StatusInternalServerError, "request_build_failed", err)
		return
	}
	uReq.Header = stripHopByHop(req.Header)
	if req.Method == http.MethodGet || req.Method == http.MethodHead {
		uReq.Body = nil
	}

	resp, err := r.doUpstreamWithRetry(req.Context(), rt.BackendURL, req.Method, req.URL.RequestURI(), uReq.Header, body)
	if err != nil {
		sendRouterErr(w, http.StatusBadGateway, "upstream_error", err)
		return
	}
	defer resp.Body.Close()

	copyHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func (r *Router) doUpstreamWithRetry(
	ctx context.Context,
	backendURL, method, requestURI string,
	headers http.Header,
	body []byte,
) (*http.Response, error) {
	backoff := 250 * time.Millisecond
	var lastResp *http.Response

	for attempt := 0; attempt <= upstream503MaxRetries; attempt++ {
		upstreamURL := strings.TrimRight(backendURL, "/") + requestURI
		var reader io.Reader = bytes.NewReader(body)
		if method == http.MethodGet || method == http.MethodHead {
			reader = nil
		}

		uReq, err := http.NewRequestWithContext(ctx, method, upstreamURL, reader)
		if err != nil {
			return nil, err
		}
		uReq.Header = headers.Clone()

		resp, err := r.client.Do(uReq)
		if err != nil {
			return nil, err
		}

		// Retry brief upstream 503 windows (e.g. model loading/restarting).
		if resp.StatusCode != http.StatusServiceUnavailable || attempt == upstream503MaxRetries {
			return resp, nil
		}

		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		lastResp = resp

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(backoff):
		}
		backoff *= 2
	}

	if lastResp != nil {
		return lastResp, nil
	}
	return nil, errors.New("upstream retry failed")
}

func (r *Router) ensureModelAvailable(ctx context.Context, rt RouteConfig) error {
	if rt.ModelFilePath == "" {
		return nil
	}
	if fileExists(rt.ModelFilePath) {
		return nil
	}
	if rt.HuggingFace == nil {
		return fmt.Errorf("model file missing: %s", rt.ModelFilePath)
	}

	mu := r.getDownloadLock(rt.ID)
	mu.Lock()
	defer mu.Unlock()

	if fileExists(rt.ModelFilePath) {
		return nil
	}

	if err := downloadFromHF(ctx, *rt.HuggingFace); err != nil {
		return err
	}
	if !fileExists(rt.ModelFilePath) && rt.HuggingFace.TargetPath != rt.ModelFilePath {
		if fileExists(rt.HuggingFace.TargetPath) {
			if err := os.MkdirAll(filepath.Dir(rt.ModelFilePath), 0o755); err != nil {
				return err
			}
			if err := os.Rename(rt.HuggingFace.TargetPath, rt.ModelFilePath); err != nil {
				return err
			}
		}
	}
	if !fileExists(rt.ModelFilePath) {
		return fmt.Errorf("download finished but model file still missing: %s", rt.ModelFilePath)
	}
	return nil
}

func (r *Router) getDownloadLock(modelID string) *sync.Mutex {
	r.dlMu.Lock()
	defer r.dlMu.Unlock()
	if m, ok := r.dlLocks[modelID]; ok {
		return m
	}
	m := &sync.Mutex{}
	r.dlLocks[modelID] = m
	return m
}

func downloadFromHF(ctx context.Context, hf HFConfig) error {
	target := hf.TargetPath
	if target == "" {
		return errors.New("huggingface target_path/model_file_path is required")
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return err
	}

	u := hfDownloadURL(hf)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	if token := strings.TrimSpace(os.Getenv(hf.TokenEnv)); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	client := &http.Client{Timeout: 0}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
			return fmt.Errorf("hf auth required: %s", resp.Status)
		}
		return fmt.Errorf("hf download failed: %s", resp.Status)
	}

	tmp := fmt.Sprintf("%s.part.%d", target, time.Now().UnixNano())
	f, err := os.OpenFile(tmp, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}

	n, copyErr := io.Copy(f, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		_ = os.Remove(tmp)
		return copyErr
	}
	if closeErr != nil {
		_ = os.Remove(tmp)
		return closeErr
	}
	if n == 0 {
		_ = os.Remove(tmp)
		return errors.New("downloaded file is empty")
	}
	if err := os.Rename(tmp, target); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return nil
}

func hfDownloadURL(hf HFConfig) string {
	// Repo must preserve owner/repo path separator while encoding each segment.
	repo := encodePathPreserveSlash(hf.Repo)
	rev := encodePathPreserveSlash(hf.Revision)
	file := encodePathPreserveSlash(hf.Filename)
	return fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s?download=true", repo, rev, file)
}

func encodePathPreserveSlash(v string) string {
	parts := strings.Split(v, "/")
	for i, p := range parts {
		parts[i] = url.PathEscape(p)
	}
	return strings.Join(parts, "/")
}

func extractModelID(contentType string, body []byte) string {
	if !strings.Contains(strings.ToLower(contentType), "application/json") || len(body) == 0 {
		return ""
	}
	var p map[string]any
	if err := json.Unmarshal(body, &p); err != nil {
		return ""
	}
	if m, ok := p["model"].(string); ok {
		return m
	}
	return ""
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !st.IsDir()
}

func stripHopByHop(h http.Header) http.Header {
	out := make(http.Header)
	for k, vv := range h {
		lk := strings.ToLower(k)
		switch lk {
		case "host", "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "content-length":
			continue
		}
		for _, v := range vv {
			out.Add(k, v)
		}
	}
	return out
}

func copyHeaders(dst, src http.Header) {
	for k, vv := range src {
		lk := strings.ToLower(k)
		if lk == "transfer-encoding" || lk == "connection" {
			continue
		}
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func sendJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func sendRouterErr(w http.ResponseWriter, status int, typ string, err error) {
	sendJSON(w, status, map[string]any{
		"error": map[string]any{
			"type":    typ,
			"message": err.Error(),
		},
	})
}
