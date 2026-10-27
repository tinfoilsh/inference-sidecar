package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"testing"
)

const (
	deltaEvent    = `data: {"id":"chunk-1","choices":[{"delta":{"content":"hi"}}]}`
	countsEvent   = `data: {"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":5}}}`
	doneEvent     = "data: [DONE]"
	expectedUsage = "prompt=11,completion=3,total=14,cached_prompt_tokens=5,uncached_prompt_tokens=6"
)

func startProxy(t *testing.T, upstream http.Handler) *httptest.Server {
	t.Helper()

	origin := httptest.NewServer(upstream)
	t.Cleanup(origin.Close)

	target, err := url.Parse(origin.URL)
	if err != nil {
		t.Fatalf("cannot parse upstream URL: %v", err)
	}

	proxy := httptest.NewServer(newHandler(target, true))
	t.Cleanup(proxy.Close)
	return proxy
}

func streamOf(events ...string) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		for _, event := range events {
			if _, err := io.WriteString(writer, event+"\n\n"); err != nil {
				return
			}
			writer.(http.Flusher).Flush()
		}
	})
}

func post(t *testing.T, proxy *httptest.Server, body string, wantMetrics bool) (*http.Response, string) {
	t.Helper()

	request, err := http.NewRequest(http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("cannot build request: %v", err)
	}
	if wantMetrics {
		request.Header.Set(usageRequestHeader, "true")
	}

	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	t.Cleanup(func() { response.Body.Close() })

	payload, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("cannot read response: %v", err)
	}
	return response, string(payload)
}

func firstDelta(t *testing.T, payload string) map[string]any {
	t.Helper()

	line, _, found := strings.Cut(payload, "\n")
	if !found || !strings.HasPrefix(line, "data: ") {
		t.Fatalf("no leading data line in %q", payload)
	}
	document := decode([]byte(strings.TrimPrefix(line, "data: ")))
	choices, ok := document["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("no choices in %q", line)
	}
	delta, ok := choices[0].(map[string]any)["delta"].(map[string]any)
	if !ok {
		t.Fatalf("no delta in %q", line)
	}
	return delta
}

func TestStreamingMeteredDropsUnrequestedCounts(t *testing.T) {
	proxy := startProxy(t, streamOf(deltaEvent, countsEvent, doneEvent))
	response, payload := post(t, proxy, `{"model":"gpt-oss-120b","stream":true}`, true)

	if strings.Contains(payload, `"usage"`) {
		t.Errorf("counts-only chunk reached a caller that did not ask for it: %q", payload)
	}
	if !strings.HasSuffix(payload, doneEvent+"\n\n") {
		t.Errorf("terminator not preserved verbatim: %q", payload)
	}

	delta := firstDelta(t, payload)
	if padding, _ := delta["p"].(string); len(padding) < minPaddingLength {
		t.Errorf("delta was not padded: %v", delta)
	}
	if delta["content"] != "hi" {
		t.Errorf("delta content was rewritten: %v", delta)
	}

	if got := response.Trailer.Get(usageResponseHeader); got != expectedUsage {
		t.Errorf("trailer = %q, want %q", got, expectedUsage)
	}
}

func TestStreamingMeteredKeepsRequestedCounts(t *testing.T) {
	proxy := startProxy(t, streamOf(deltaEvent, countsEvent, doneEvent))
	response, payload := post(t, proxy, `{"stream":true,"stream_options":{"include_usage":true}}`, true)

	if !strings.Contains(payload, `"total_tokens":14`) {
		t.Errorf("counts-only chunk was dropped for a caller that asked: %q", payload)
	}
	if got := response.Trailer.Get(usageResponseHeader); got != expectedUsage {
		t.Errorf("trailer = %q, want %q", got, expectedUsage)
	}
}

func TestStreamingUnmeteredStillPads(t *testing.T) {
	proxy := startProxy(t, streamOf(deltaEvent, countsEvent, doneEvent))
	response, payload := post(t, proxy, `{"stream":true}`, false)

	delta := firstDelta(t, payload)
	if padding, _ := delta["p"].(string); len(padding) < minPaddingLength {
		t.Errorf("delta was not padded: %v", delta)
	}
	if response.Header.Get("Trailer") != "" {
		t.Errorf("unmetered response announced a trailer: %q", response.Header.Get("Trailer"))
	}
	if got := response.Trailer.Get(usageResponseHeader); got != "" {
		t.Errorf("unmetered response carried counts: %q", got)
	}
}

func TestNonStreamingMeteredReportsOnHeader(t *testing.T) {
	const document = `{"id":"resp-1","usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14,"prompt_tokens_details":{"cached_tokens":5}}}`

	proxy := startProxy(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		writer.Header().Set("Content-Type", "application/json")
		io.WriteString(writer, document)
	}))
	response, payload := post(t, proxy, `{"model":"gpt-oss-120b"}`, true)

	if got := response.Header.Get(usageResponseHeader); got != expectedUsage {
		t.Errorf("header = %q, want %q", got, expectedUsage)
	}
	if payload != document {
		t.Errorf("body = %q, want %q", payload, document)
	}
	if response.ContentLength != int64(len(document)) {
		t.Errorf("Content-Length = %d, want %d", response.ContentLength, len(document))
	}
}

func TestLastPortFlag(t *testing.T) {
	cases := []struct {
		name string
		args []string
		want string
	}{
		{"separate value", []string{"vllm", "serve", "--port", "8001"}, "8001"},
		{"inline value", []string{"vllm", "serve", "--port=8001"}, "8001"},
		{"last wins", []string{"vllm", "--port", "8001", "--port=9002"}, "9002"},
		{"absent", []string{"vllm", "serve", "--model", "/tinfoil/mpk/mpk-0"}, defaultListenPort},
		{"dangling", []string{"vllm", "serve", "--port"}, defaultListenPort},
	}

	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			if got := lastPortFlag(test.args, defaultListenPort); got != test.want {
				t.Errorf("lastPortFlag(%q) = %q, want %q", test.args, got, test.want)
			}
		})
	}
}

func TestUpstreamCommandForcesLoopback(t *testing.T) {
	args := []string{"vllm", "serve", "--speculative-config", `{"method":"dspark","num_speculative_tokens":7}`, "--port", "8001"}
	got := upstreamCommand(append([]string{}, args...), defaultUpstreamPort)

	want := append(append([]string{}, args...), "--host", upstreamHost, "--port", defaultUpstreamPort)
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Errorf("upstreamCommand() = %q, want %q", got, want)
	}
}

func TestLargeRequestBodyReachesUpstream(t *testing.T) {
	body := `{"pad":"` + strings.Repeat("a", 9<<20) + `"}`

	proxy := startProxy(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		received, err := io.Copy(io.Discard, request.Body)
		if err != nil {
			t.Errorf("upstream read failed: %v", err)
		}
		writer.Header().Set("Content-Type", "application/json")
		io.WriteString(writer, `{"received":`+strconv.FormatInt(received, 10)+`}`)
	}))
	_, payload := post(t, proxy, body, true)

	var document struct {
		Received int `json:"received"`
	}
	if err := json.Unmarshal([]byte(payload), &document); err != nil {
		t.Fatalf("cannot decode upstream reply %q: %v", payload, err)
	}
	if document.Received != len(body) {
		t.Errorf("upstream received %d bytes, want %d", document.Received, len(body))
	}
}
