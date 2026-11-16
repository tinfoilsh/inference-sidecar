package main

// developerRole renames the `developer` role to `system`.
//
// Reproduced on 2026-11-15 against production: Kimi K3.1 Pro rejects
// `developer` with a 400 ("Unknown message role 'developer' at index 0"),
// and DeepSeek V4.1 Flash accepts it but ignores the message. GLM 5.3,
// Inkling 2 Small, Gemma 4, Llama 3.3 and gpt-oss all honour it. gpt-oss in
// particular treats `developer` and `system` as distinct roles, which is why
// this rule is opt-in per model.
//
// Only the role value changes: message-local `tools` and every other field
// survive, and system messages are not consolidated, because vLLM's own
// consolidation joins content with newlines and mutates the prompt.
var developerRole = quirk{
	name:   "developer-role",
	paths:  []string{"/v1/chat/completions", "/v1/responses", "/tokenize"},
	marker: []byte(`"developer"`),
	rewrite: func(path string, at jsonPath, value string) (string, bool) {
		// On /v1/responses `input` may be a bare string; that has no role.
		if value == "developer" && at.is(messageListKey(path), "[]", "role") {
			return "system", true
		}
		return value, false
	},
}
