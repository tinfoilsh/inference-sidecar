package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
)

func TestSelectQuirks(t *testing.T) {
	cases := []struct {
		spec string
		want string
	}{
		{"", "none"},
		{"none", "none"},
		{"developer-role", "developer-role"},
		{" developer-role ", "developer-role"},
		{"developer-role,developer-role", "developer-role"},
	}
	for _, test := range cases {
		t.Run(test.spec, func(t *testing.T) {
			rules, err := selectQuirks(test.spec)
			if err != nil {
				t.Fatalf("selectQuirks(%q): %v", test.spec, err)
			}
			if got := quirkNames(rules); got != test.want {
				t.Errorf("selectQuirks(%q) = %q, want %q", test.spec, got, test.want)
			}
		})
	}
}

func TestSelectQuirksRejectsUnknownNames(t *testing.T) {
	for _, spec := range []string{"developer_role", "all", "-developer-role", "developer-role,"} {
		if _, err := selectQuirks(spec); err == nil {
			t.Errorf("selectQuirks(%q) accepted an unknown name", spec)
		}
	}
}

func TestRewriteLeavesBodyAloneWhenNothingToDo(t *testing.T) {
	tests := []struct {
		name string
		path string
		body string
	}{
		{"unrelated path", "/v1/embeddings", `{"messages":[{"role":"developer","content":"x"}]}`},
		{"no developer role present", "/v1/chat/completions", `{"messages":[{"role":"user","content":"x"}]}`},
		{"developer appears only as content", "/v1/chat/completions", `{"messages":[{"role":"user","content":"the word \"developer\" here"}]}`},
		{"malformed JSON", "/v1/chat/completions", `{"messages":[{"role":"developer"`},
		{"trailing garbage", "/v1/chat/completions", `{"messages":[{"role":"developer","content":"x"}]} {}`},
		{"responses input as a bare string", "/v1/responses", `{"input":"hello","instructions":"you are a \"developer\" assistant"}`},
		{"messages is not an array", "/v1/chat/completions", `{"messages":"developer"}`},
		{"trailing close brace", "/v1/chat/completions", `{"messages":[{"role":"developer"}]}}`},
		{"trailing close bracket", "/v1/chat/completions", `{"messages":[{"role":"developer"}]}]`},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, count := rewriteRequestBody(tc.path, []byte(tc.body), quirks)
			if count != 0 {
				t.Errorf("rewrite count = %d, want 0", count)
			}
			if string(out) != tc.body {
				t.Errorf("body was modified:\n got %s\nwant %s", out, tc.body)
			}
		})
	}
}

// Rules are opt-in: an unset SIDECAR_QUIRKS must forward bodies untouched.
func TestRewriteRunsNothingByDefault(t *testing.T) {
	body := `{"messages":[{"role":"developer","content":"x"}]}`
	rules, err := selectQuirks("")
	if err != nil {
		t.Fatal(err)
	}
	if out, count := rewriteRequestBody("/v1/chat/completions", []byte(body), rules); count != 0 || string(out) != body {
		t.Errorf("a rule ran without opting in: count=%d body=%s", count, out)
	}
}

// A float64 round-trip would silently rewrite a large seed; UseNumber stops it.
func TestRewritePreservesNumericLiterals(t *testing.T) {
	body := `{"seed":12345678901234567,"temperature":0.70,"messages":[{"role":"developer","content":"x"}]}`

	out, count := rewriteRequestBody("/v1/chat/completions", []byte(body), quirks)
	if count != 1 {
		t.Fatalf("rewrite count = %d, want 1", count)
	}
	if !bytes.Contains(out, []byte(`12345678901234567`)) {
		t.Errorf("seed lost precision: %s", out)
	}
	if !bytes.Contains(out, []byte(`0.70`)) {
		t.Errorf("temperature literal was reformatted: %s", out)
	}
}

func TestRewriteDoesNotEscapeHTML(t *testing.T) {
	body := `{"messages":[{"role":"developer","content":"if a < b && c > d"}]}`

	out, count := rewriteRequestBody("/v1/chat/completions", []byte(body), quirks)
	if count != 1 {
		t.Fatalf("rewrite count = %d, want 1", count)
	}
	if !bytes.Contains(out, []byte("a < b && c > d")) {
		t.Errorf("prompt text was escaped: %s", out)
	}
}

func TestProxyForwardsRewrittenBodyWithCorrectLength(t *testing.T) {
	var received []byte
	var declared int64
	proxy := startProxy(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		declared = request.ContentLength
		received, _ = io.ReadAll(request.Body)
		writer.Header().Set("Content-Type", "application/json")
		io.WriteString(writer, `{}`)
	}), developerRole)

	body := `{"model":"kimi-k3-1-pro","messages":[{"role":"developer","content":"x"}]}`
	request, err := http.NewRequest(http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json; charset=utf-8")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()

	if bytes.Contains(received, []byte(`"developer"`)) {
		t.Errorf("upstream saw the developer role: %s", received)
	}
	if !bytes.Contains(received, []byte(`"system"`)) {
		t.Errorf("upstream did not see the rewrite: %s", received)
	}
	if declared != int64(len(received)) {
		t.Errorf("upstream Content-Length = %d, body was %d bytes", declared, len(received))
	}
}

func TestProxyLeavesNonJSONBodiesAlone(t *testing.T) {
	var received []byte
	proxy := startProxy(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		received, _ = io.ReadAll(request.Body)
		io.WriteString(writer, "ok")
	}), developerRole)

	body := `{"messages":[{"role":"developer","content":"x"}]}`
	request, _ := http.NewRequest(http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(body))
	request.Header.Set("Content-Type", "text/plain")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if string(received) != body {
		t.Errorf("non-JSON body was rewritten: %s", received)
	}
}

// A rewritten stream request must still have its usage intent recorded.
func TestRewrittenStreamingRequestStillDropsUnrequestedCounts(t *testing.T) {
	proxy := startProxy(t, streamOf(deltaEvent, countsEvent, doneEvent), developerRole)
	response, payload := post(t, proxy, `{"stream":true,"messages":[{"role":"developer","content":"x"}]}`, true)

	if strings.Contains(payload, `"usage"`) {
		t.Errorf("counts-only chunk leaked after a rewrite: %q", payload)
	}
	if got := response.Trailer.Get(usageResponseHeader); got != expectedUsage {
		t.Errorf("trailer = %q, want %q", got, expectedUsage)
	}
}

// An oversize body must still be forwarded in full, and never rewritten.
func TestOversizeRequestBodyForwardedIntact(t *testing.T) {
	filler := strings.Repeat("a", maxRequestBytes+16)
	body := `{"messages":[{"role":"developer","content":"` + filler + `"}]}`

	var seenDeveloper bool
	proxy := startProxy(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		received, err := io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("upstream read failed: %v", err)
		}
		seenDeveloper = bytes.Contains(received, []byte(`"developer"`))
		writer.Header().Set("Content-Type", "application/json")
		io.WriteString(writer, `{"received":`+strconv.Itoa(len(received))+`}`)
	}), developerRole)
	_, payload := post(t, proxy, body, false)

	var document struct {
		Received int `json:"received"`
	}
	if err := json.Unmarshal([]byte(payload), &document); err != nil {
		t.Fatalf("cannot decode upstream reply %q: %v", payload, err)
	}
	if document.Received != len(body) {
		t.Errorf("upstream received %d bytes, want %d", document.Received, len(body))
	}
	if !seenDeveloper {
		t.Error("oversize body was rewritten instead of forwarded intact")
	}
}

func TestWalkStringsReportsLocations(t *testing.T) {
	body := `{"a":"x","b":[{"c":"y"},"z"],"d":{"e":["w"]},"n":1}`
	var got []string
	err := walkStrings([]byte(body), func(at jsonPath, start, end int, value string) {
		got = append(got, strings.Join(at, ".")+"="+value+"@"+body[start:end])
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []string{`a=x@"x"`, `b.[].c=y@"y"`, `b.[]=z@"z"`, `d.e.[]=w@"w"`}
	if strings.Join(got, " ") != strings.Join(want, " ") {
		t.Errorf("got %v, want %v", got, want)
	}
}

// With SIDECAR_QUIRKS unset the proxy must forward a developer body exactly
// as main did before rules existed.
func TestProxyWithoutRulesForwardsBodyUntouched(t *testing.T) {
	var received []byte
	var declared int64
	proxy := startProxy(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		declared = request.ContentLength
		received, _ = io.ReadAll(request.Body)
		io.WriteString(writer, "{}")
	}))

	body := `{"messages":[{"role":"developer","content":"x"}]}`
	response, err := http.Post(proxy.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if string(received) != body || declared != int64(len(body)) {
		t.Errorf("upstream got %d bytes declared %d: %s", len(received), declared, received)
	}
}

// A chunked upload has no Content-Length to fix up; the rewrite must still
// arrive whole.
func TestProxyRewritesChunkedBody(t *testing.T) {
	var received []byte
	proxy := startProxy(t, http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		received, _ = io.ReadAll(request.Body)
		io.WriteString(writer, "{}")
	}), developerRole)

	body := `{"messages":[{"role":"developer","content":"x"}]}`
	// A reader of unknown length makes the client send Transfer-Encoding: chunked.
	request, err := http.NewRequest(http.MethodPost, proxy.URL+"/v1/chat/completions", io.MultiReader(strings.NewReader(body)))
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	if want := strings.Replace(body, "developer", "system", 1); string(received) != want {
		t.Errorf("upstream got %s, want %s", received, want)
	}
}
