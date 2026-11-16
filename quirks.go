package main

// Request-side compatibility rules ("quirks").
//
// Tinfoil advertises an OpenAI-compatible endpoint. vLLM, and the
// model-specific renderers behind it, diverge from that contract in small
// ways no client can predict from outside the enclave. The sidecar is the
// last hop before vLLM and the first place that is both inside the trust
// boundary and specific to inference, so the repairs live here rather than
// in the shim (platform code every workload shares) or the router (which
// cannot read a sealed body).
//
// Every rule is a lossless rewrite with exactly one correct answer. A rule
// that needs a product decision belongs in /v1/models metadata, not here.
// Rules replace individual string values and the replacements are spliced
// into the original bytes, so key order, whitespace, numbers and escapes
// everywhere else reach vLLM exactly as the client sent them; chat
// templates render tool schemas in key order, so a re-encode would change
// the prompt. A body that is not valid JSON is forwarded untouched; the
// sidecar never rejects a request vLLM would have accepted.
//
// Rules are off unless a model opts in with SIDECAR_QUIRKS in its
// tinfoil-config.yml. A rewrite that fixes one model can change the prompt
// of another that handles the original correctly, so a rule only runs where
// the bug it repairs was reproduced.
//
// Adding a rule:
//  1. Write quirk_<name>.go with a quirk value and its rewrite function.
//  2. Add the value to the quirks registry below.
//  3. Write quirk_<name>_test.go, including bodies the rule must not touch.
//  4. Opt the affected models in via SIDECAR_QUIRKS.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
)

type quirk struct {
	name string
	// paths the rule applies to, without a trailing slash.
	paths []string
	// marker is a cheap pre-filter: bodies without the literal are never
	// parsed, so the common path stays byte-exact.
	marker []byte
	// rewrite is offered every string value in the body with its location
	// and returns the replacement and true to change it.
	rewrite func(path string, at jsonPath, value string) (string, bool)
}

// jsonPath locates a value in a body: the object keys leading to it, with
// "[]" standing for any array element.
type jsonPath []string

func (p jsonPath) is(want ...string) bool {
	return slices.Equal(p, want)
}

// quirks is every rule the sidecar knows, in the order they are applied.
var quirks = []quirk{
	developerRole,
}

// selectQuirks parses SIDECAR_QUIRKS: empty or "none" disables every rule,
// otherwise it is a comma-separated list of rule names. Unknown names are an
// error: a typo in measured config should fail boot, not silently run with
// the wrong set. Rules always run in registry order, whatever the list order.
func selectQuirks(spec string) ([]quirk, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || spec == "none" {
		return nil, nil
	}

	enabled := map[string]bool{}
	for name := range strings.SplitSeq(spec, ",") {
		name = strings.TrimSpace(name)
		if !slices.ContainsFunc(quirks, func(rule quirk) bool { return rule.name == name }) {
			return nil, fmt.Errorf("unknown quirk %q", name)
		}
		enabled[name] = true
	}

	var selected []quirk
	for _, rule := range quirks {
		if enabled[rule.name] {
			selected = append(selected, rule)
		}
	}
	return selected, nil
}

func quirkNames(rules []quirk) string {
	if len(rules) == 0 {
		return "none"
	}
	names := make([]string, 0, len(rules))
	for _, rule := range rules {
		names = append(names, rule.name)
	}
	return strings.Join(names, ",")
}

// claimsPath reports whether any rule applies to path, so the proxy only
// buffers whole bodies for requests a rule might rewrite.
func claimsPath(rules []quirk, path string) bool {
	path = strings.TrimSuffix(path, "/")
	return slices.ContainsFunc(rules, func(rule quirk) bool { return slices.Contains(rule.paths, path) })
}

type splice struct {
	start, end  int
	replacement []byte
}

// rewriteRequestBody applies every enabled rule that claims path to body and
// reports how many values changed. It returns body itself whenever nothing
// applies or the body is not a single valid JSON value.
func rewriteRequestBody(path string, body []byte, rules []quirk) ([]byte, int) {
	path = strings.TrimSuffix(path, "/")
	var applicable []quirk
	for _, rule := range rules {
		if slices.Contains(rule.paths, path) && bytes.Contains(body, rule.marker) {
			applicable = append(applicable, rule)
		}
	}
	// Valid also rejects trailing content, so a body is only rewritten when
	// it is exactly the one value vLLM will parse.
	if len(applicable) == 0 || !json.Valid(body) {
		return body, 0
	}

	var splices []splice
	err := walkStrings(body, func(at jsonPath, start, end int, value string) {
		original := value
		for _, rule := range applicable {
			if replacement, ok := rule.rewrite(path, at, value); ok {
				value = replacement
			}
		}
		if value != original {
			splices = append(splices, splice{start, end, encodeString(value)})
		}
	})
	if err != nil || len(splices) == 0 {
		return body, 0
	}

	out := make([]byte, 0, len(body))
	last := 0
	for _, edit := range splices {
		out = append(out, body[last:edit.start]...)
		out = append(out, edit.replacement...)
		last = edit.end
	}
	return append(out, body[last:]...), len(splices)
}

// walkStrings calls visit for every string value in body (not object keys)
// with its location and the byte range of its quoted literal.
func walkStrings(body []byte, visit func(at jsonPath, start, end int, value string)) error {
	type frame struct {
		object    bool
		expectKey bool
		element   string
	}
	var stack []*frame
	location := func() jsonPath {
		at := make(jsonPath, len(stack))
		for index, f := range stack {
			at[index] = f.element
		}
		return at
	}
	// valueDone marks the end of a value, so an enclosing object's next
	// string token is read as a key.
	valueDone := func() {
		if len(stack) > 0 && stack[len(stack)-1].object {
			stack[len(stack)-1].expectKey = true
		}
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	for {
		before := int(decoder.InputOffset())
		token, err := decoder.Token()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		if len(stack) > 0 {
			if top := stack[len(stack)-1]; top.object && top.expectKey {
				if key, ok := token.(string); ok {
					top.element, top.expectKey = key, false
					continue
				}
			}
		}

		switch value := token.(type) {
		case json.Delim:
			switch value {
			case '{':
				stack = append(stack, &frame{object: true, expectKey: true})
			case '[':
				stack = append(stack, &frame{element: "[]"})
			default:
				stack = stack[:len(stack)-1]
				valueDone()
			}
		case string:
			end := int(decoder.InputOffset())
			// Token consumes the separators before a value; the literal
			// starts at the first quote after them.
			start := before + bytes.IndexByte(body[before:end], '"')
			visit(location(), start, end, value)
			valueDone()
		default:
			valueDone()
		}
	}
}

// encodeString quotes value as a JSON string, leaving <, > and & as
// themselves so prompt text round-trips.
func encodeString(value string) []byte {
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)
	encoder.Encode(value)
	return bytes.TrimRight(out.Bytes(), "\n")
}

// messageListKey reports which top-level field holds role-bearing items.
// /tokenize is included because it applies the same chat template.
func messageListKey(path string) string {
	if path == "/v1/responses" {
		// Only input items have roles; `instructions` is roleless.
		return "input"
	}
	return "messages"
}
