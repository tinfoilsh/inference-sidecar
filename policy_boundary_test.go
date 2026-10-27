package main

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/tinfoilsh/inference-sidecar/internal/key"
)

type unreadBody struct{ reads int }

func (b *unreadBody) Read([]byte) (int, error) {
	b.reads++
	return 0, errors.New("body read before authentication")
}
func (*unreadBody) Close() error { return nil }

func TestRejectedCredentialDoesNotReadBody(t *testing.T) {
	upstream, _ := url.Parse("http://127.0.0.1:1")
	for _, authorization := range []string{"", "Bearer bad"} {
		body := &unreadBody{}
		handler := withPolicy(&Policy{}, "deployment.example", &fakeValidator{err: &key.ValidationError{StatusCode: 401}}, nil, newHandler(upstream, true))
		request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", body)
		request.Header.Set("Authorization", authorization)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != 401 || body.reads != 0 {
			t.Fatalf("status=%d, body reads=%d", response.Code, body.reads)
		}
	}
}

func TestOriginalHostTrustBoundary(t *testing.T) {
	for _, tc := range []struct{ name, peer, host, forwarded, want string }{
		{"direct", "192.0.2.1:1234", "Model.Example:443", "admin.example", "model.example"},
		{"untrusted localhost", "192.0.2.1:1234", "localhost", "admin.example", "localhost"},
		{"old shim", "172.31.255.1:1234", "localhost", "Model.Example:443", "model.example"},
		{"new shim", "172.31.255.1:1234", "Model.Example:443", "admin.example", "model.example"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			validator := &fakeValidator{}
			var forwarded string
			handler := withPolicy(&Policy{TrustedProxies: []string{"172.31.255.1"}}, "DEPLOYMENT.EXAMPLE", validator, nil, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				forwarded = r.Header.Get("X-Forwarded-Host")
			}))
			request := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			request.Host, request.RemoteAddr = tc.host, tc.peer
			request.Header.Set("Authorization", "Bearer credential")
			request.Header.Set("X-Forwarded-Host", tc.forwarded)
			request.Header.Set("Forwarded", "host=admin.example")
			handler.ServeHTTP(httptest.NewRecorder(), request)
			if len(validator.calls) != 1 || validator.calls[0].RequestedHost != tc.want || validator.calls[0].Domain != "deployment.example" {
				t.Fatalf("control-plane request = %+v", validator.calls)
			}
			if forwarded != request.Host || request.Header.Get("Forwarded") != "" {
				t.Fatalf("untrusted forwarding metadata escaped: %v", request.Header)
			}
		})
	}
}

func TestPolicyCORSAndExplicitUpstream(t *testing.T) {
	const event = "data: {\"jsonrpc\":\"2.0\",\"result\":{}}\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Access-Control-Allow-Origin", "https://untrusted.example")
		io.WriteString(w, event)
	}))
	defer upstream.Close()
	target, _ := url.Parse(upstream.URL)
	validator := &fakeValidator{}
	paths := []string{"/mcp"}
	handler := withPolicy(&Policy{OriginDomains: []string{"https://client.example"}, AuthenticatedEndpoints: &paths}, "deployment.example", validator, nil, newHandler(target, false))
	for _, tc := range []struct {
		method, origin string
		status         int
	}{
		{http.MethodOptions, "https://client.example", http.StatusNoContent},
		{http.MethodPost, "https://bad.example", http.StatusForbidden},
		{http.MethodPost, "https://client.example", http.StatusOK},
	} {
		request := httptest.NewRequest(tc.method, "/mcp", strings.NewReader("{}"))
		request.Header.Set("Origin", tc.origin)
		request.Header.Set("Authorization", "Bearer valid")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != tc.status {
			t.Fatalf("status=%d: %s", response.Code, response.Body.String())
		}
		if tc.status == 200 && (response.Body.String() != event || response.Header().Get("Access-Control-Allow-Origin") != tc.origin) {
			t.Fatalf("MCP body or policy origin changed: %s %v", response.Body.String(), response.Header())
		}
	}
	if len(validator.calls) != 1 {
		t.Fatalf("CORS must run before auth: %+v", validator.calls)
	}
}

func TestPolicyRequiresExplicitConfiguration(t *testing.T) {
	for _, raw := range []string{"", `{}`, `{"version":2,"authenticated":false}`, `{"version":1,"authenticated":true}`, `{"version":1,"authenticated":false,"unknown":true}`, `{"version":1,"authenticated":false} {}`} {
		if _, err := parsePolicy(raw, "deployment.example"); err == nil {
			t.Fatalf("accepted %q", raw)
		}
	}
	if _, err := parsePolicy(`{"version":1,"authenticated":true,"control-plane":"https://api.example"}`, ""); err == nil {
		t.Fatal("accepted missing domain")
	}
	if _, err := parsePolicy(`{"version":1,"authenticated":false}`, ""); err != nil {
		t.Fatal(err)
	}
}
