package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func TestDeveloperRoleRewrite(t *testing.T) {
	tests := []struct {
		name      string
		path      string
		body      string
		wantRoles []string
		wantCount int
	}{
		{
			name:      "chat completions leading developer message",
			path:      "/v1/chat/completions",
			body:      `{"model":"kimi-k3-1-pro","messages":[{"role":"developer","content":"be terse"},{"role":"user","content":"hi"}]}`,
			wantRoles: []string{"system", "user"},
			wantCount: 1,
		},
		{
			name:      "tokenize shares the chat template",
			path:      "/tokenize",
			body:      `{"model":"kimi-k3-1-pro","messages":[{"role":"developer","content":"x"}]}`,
			wantRoles: []string{"system"},
			wantCount: 1,
		},
		{
			name:      "responses input items",
			path:      "/v1/responses",
			body:      `{"model":"kimi-k3-1-pro","input":[{"role":"developer","content":"x"},{"role":"user","content":"y"}]}`,
			wantRoles: []string{"system", "user"},
			wantCount: 1,
		},
		{
			name:      "trailing slash still matches",
			path:      "/v1/chat/completions/",
			body:      `{"messages":[{"role":"developer","content":"x"}]}`,
			wantRoles: []string{"system"},
			wantCount: 1,
		},
		{
			name:      "every developer message is rewritten, not just the first",
			path:      "/v1/chat/completions",
			body:      `{"messages":[{"role":"developer","content":"a"},{"role":"user","content":"b"},{"role":"developer","content":"c"}]}`,
			wantRoles: []string{"system", "user", "system"},
			wantCount: 2,
		},
		{
			name:      "already-system bodies are untouched",
			path:      "/v1/chat/completions",
			body:      `{"messages":[{"role":"system","content":"x"},{"role":"user","content":"y"}]}`,
			wantRoles: []string{"system", "user"},
			wantCount: 0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, count := rewriteRequestBody(tc.path, []byte(tc.body), quirks)
			if count != tc.wantCount {
				t.Fatalf("rewrite count = %d, want %d", count, tc.wantCount)
			}

			listKey := "messages"
			if strings.HasPrefix(tc.path, "/v1/responses") {
				listKey = "input"
			}
			var parsed map[string]json.RawMessage
			if err := json.Unmarshal(out, &parsed); err != nil {
				t.Fatalf("result is not valid JSON: %v", err)
			}
			var items []map[string]any
			if err := json.Unmarshal(parsed[listKey], &items); err != nil {
				t.Fatalf("%s is not a message list: %v", listKey, err)
			}
			if len(items) != len(tc.wantRoles) {
				t.Fatalf("got %d messages, want %d", len(items), len(tc.wantRoles))
			}
			for i, want := range tc.wantRoles {
				if got, _ := items[i]["role"].(string); got != want {
					t.Errorf("message %d role = %q, want %q", i, got, want)
				}
			}
			if bytes.Contains(out, []byte(`"developer"`)) {
				t.Error("result still contains the developer role")
			}
		})
	}
}

// Message-local `tools` must survive the rename.
func TestDeveloperRoleRewritePreservesMessageTools(t *testing.T) {
	body := `{"messages":[{"role":"developer","content":"x","tools":[{"type":"function","function":{"name":"search"}}]}]}`

	out, count := rewriteRequestBody("/v1/chat/completions", []byte(body), quirks)
	if count != 1 {
		t.Fatalf("rewrite count = %d, want 1", count)
	}

	var parsed struct {
		Messages []struct {
			Role  string           `json:"role"`
			Tools []map[string]any `json:"tools"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}
	if parsed.Messages[0].Role != "system" {
		t.Errorf("role = %q, want system", parsed.Messages[0].Role)
	}
	if len(parsed.Messages[0].Tools) != 1 {
		t.Fatalf("message-local tools were dropped")
	}
}

// Merging system messages mutates the prompt, so positions are left alone.
func TestDeveloperRoleRewriteDoesNotConsolidateSystemMessages(t *testing.T) {
	body := `{"messages":[{"role":"developer","content":"a"},{"role":"user","content":"b"},{"role":"system","content":"c"}]}`

	out, count := rewriteRequestBody("/v1/chat/completions", []byte(body), quirks)
	if count != 1 {
		t.Fatalf("rewrite count = %d, want 1", count)
	}

	var parsed struct {
		Messages []map[string]any `json:"messages"`
	}
	if err := json.Unmarshal(out, &parsed); err != nil {
		t.Fatalf("result is not valid JSON: %v", err)
	}
	if len(parsed.Messages) != 3 {
		t.Fatalf("got %d messages, want 3: messages were merged", len(parsed.Messages))
	}
	if got, _ := parsed.Messages[1]["role"].(string); got != "user" {
		t.Errorf("message 1 role = %q, want user: order changed", got)
	}
}

// Chat templates render tool schemas in key order, so everything but the
// role must reach vLLM byte for byte: order, whitespace, numbers, escapes.
func TestDeveloperRoleRewriteChangesOnlyTheRole(t *testing.T) {
	body := `{ "model": "kimi-k3-1-pro",
  "messages": [ {"content": "a < b \ud800  ", "role" : "developer"}, {"role":"user","content":"hi"} ],
  "tools": [{"type":"function","function":{"name":"f","parameters":{"properties":{"zeta":{"type":"string"},"alpha":{"type":"number"}}}}}],
  "seed": 12345678901234567, "temperature": 0.70 }`
	want := strings.Replace(body, `"role" : "developer"`, `"role" : "system"`, 1)

	out, count := rewriteRequestBody("/v1/chat/completions", []byte(body), quirks)
	if count != 1 {
		t.Fatalf("rewrite count = %d, want 1", count)
	}
	if string(out) != want {
		t.Errorf("bytes outside the role changed:\n got %s\nwant %s", out, want)
	}
}

func TestDeveloperRoleRewriteIgnoresOtherDeveloperStrings(t *testing.T) {
	tests := []struct {
		name string
		path string
		body string
	}{
		{"as content", "/v1/chat/completions", `{"messages":[{"role":"user","content":"developer"}]}`},
		{"as a name", "/v1/chat/completions", `{"messages":[{"role":"user","name":"developer"}]}`},
		{"as a key", "/v1/chat/completions", `{"messages":[{"role":"user","developer":"role"}]}`},
		{"inside a tool schema", "/v1/chat/completions", `{"messages":[{"role":"user"}],"tools":[{"function":{"parameters":{"properties":{"role":{"enum":["developer"]}}}}}]}`},
		{"in a nested role field", "/v1/chat/completions", `{"messages":[{"role":"user","content":[{"type":"text","role":"developer"}]}]}`},
		{"messages key on responses", "/v1/responses", `{"messages":[{"role":"developer"}],"input":"x"}`},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if out, count := rewriteRequestBody(tc.path, []byte(tc.body), quirks); count != 0 || string(out) != tc.body {
				t.Errorf("body was modified: count=%d %s", count, out)
			}
		})
	}
}
