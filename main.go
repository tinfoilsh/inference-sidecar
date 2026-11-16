package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/big"
	"mime"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"sync"
	"syscall"
)

const (
	usageRequestHeader  = "X-Tinfoil-Request-Usage-Metrics"
	usageResponseHeader = "X-Tinfoil-Usage-Metrics"
	upstreamHost        = "127.0.0.1"
	defaultUpstreamPort = "18001"
	defaultListenPort   = "8000"
	maxBufferBytes      = 8 << 20
	maxRequestBytes     = 64 << 20 // largest body the quirks rewrite; larger ones pass through
	maxEventBytes       = 4 << 20
	initialEventBytes   = 64 << 10
	paddingCharset      = "abcdefghijklmnopqrstuvwxyz0123456789"
	minPaddingLength    = 4
)

var (
	eventSeparator = []byte("\n\n")
	newline        = []byte("\n")
	dataPrefix     = []byte("data:")
	dataLinePrefix = []byte("data: ")
)

func main() {
	log.SetFlags(0)
	log.SetPrefix("inference-sidecar: ")

	child := os.Args[1:]
	if len(child) == 0 {
		log.Fatal("no upstream command given")
	}

	upstreamPort := env("SIDECAR_UPSTREAM_PORT", defaultUpstreamPort)
	listenPort := env("SIDECAR_LISTEN", lastPortFlag(child, defaultListenPort))
	rules, err := selectQuirks(os.Getenv("SIDECAR_QUIRKS"))
	if err != nil {
		log.Fatalf("SIDECAR_QUIRKS: %v", err)
	}
	log.Printf("quirks enabled: %s", quirkNames(rules))
	child = upstreamCommand(child, upstreamPort)

	cmd := exec.Command(child[0], child[1:]...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	if err := cmd.Start(); err != nil {
		log.Fatalf("cannot start %s: %v", child[0], err)
	}

	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		for received := range signals {
			if err := cmd.Process.Signal(received); err != nil {
				log.Printf("cannot relay %s: %v", received, err)
			}
		}
	}()

	go serve(listenPort, upstreamPort, rules)

	if err := cmd.Wait(); cmd.ProcessState == nil {
		log.Fatalf("cannot wait for %s: %v", child[0], err)
	}
	os.Exit(exitCode(cmd.ProcessState))
}

func serve(listenPort, upstreamPort string, rules []quirk) {
	upstream := &url.URL{Scheme: "http", Host: net.JoinHostPort(upstreamHost, upstreamPort)}
	err := http.ListenAndServe(net.JoinHostPort("", listenPort), newHandler(upstream, rules))
	log.Fatalf("listener stopped: %v", err)
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func lastPortFlag(args []string, fallback string) string {
	port := fallback
	for index, arg := range args {
		switch {
		case arg == "--port" && index+1 < len(args):
			port = args[index+1]
		case strings.HasPrefix(arg, "--port="):
			port = strings.TrimPrefix(arg, "--port=")
		}
	}
	return port
}

func upstreamCommand(args []string, upstreamPort string) []string {
	return append(args, "--host", upstreamHost, "--port", upstreamPort)
}

func exitCode(state *os.ProcessState) int {
	if status, ok := state.Sys().(syscall.WaitStatus); ok && status.Signaled() {
		return 128 + int(status.Signal())
	}
	return state.ExitCode()
}

func newHandler(upstream *url.URL, rules []quirk) http.Handler {
	proxy := &httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			request.SetURL(upstream)
			request.Out.Host = request.In.Host
		},
		FlushInterval:  -1,
		ModifyResponse: modifyResponse,
	}
	return withRequestBody(proxy, rules)
}

type usageIntentKey struct{}

type requestBody struct {
	io.Reader
	io.Closer
}

// withRequestBody reads the head of each POST body to record whether the
// caller asked for the usage chunk, so modifyResponse can drop it otherwise.
// When an enabled rule claims the path it reads the whole body instead and
// applies the compatibility rules to it. A body over maxRequestBytes is
// forwarded intact and unrewritten, exactly as it arrived.
func withRequestBody(next http.Handler, rules []quirk) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			next.ServeHTTP(writer, request)
			return
		}

		rewrite := claimsPath(rules, request.URL.Path) && isJSONContentType(request.Header.Get("Content-Type"))
		limit := int64(maxBufferBytes)
		if rewrite {
			// One byte past the cap, so an unset Content-Length is still caught.
			limit = maxRequestBytes + 1
		}
		head, err := io.ReadAll(io.LimitReader(request.Body, limit))
		if err != nil {
			http.Error(writer, "cannot read request body", http.StatusBadRequest)
			return
		}

		if !rewrite || len(head) > maxRequestBytes {
			request.Body = &requestBody{
				Reader: io.MultiReader(bytes.NewReader(head), request.Body),
				Closer: request.Body,
			}
		} else if body, count := rewriteRequestBody(request.URL.Path, head, rules); count > 0 {
			request.Body = &requestBody{Reader: bytes.NewReader(body), Closer: request.Body}
			request.ContentLength = int64(len(body))
			// The inbound header would otherwise advertise the pre-rewrite length.
			request.Header.Del("Content-Length")
		} else {
			request.Body = &requestBody{Reader: bytes.NewReader(head), Closer: request.Body}
		}

		intent := context.WithValue(request.Context(), usageIntentKey{}, askedForUsageChunk(head))
		next.ServeHTTP(writer, request.WithContext(intent))
	})
}

// isJSONContentType treats a missing Content-Type as JSON, as vLLM does.
func isJSONContentType(contentType string) bool {
	if contentType == "" {
		return true
	}
	mediaType, _, err := mime.ParseMediaType(contentType)
	return err == nil && mediaType == "application/json"
}

func askedForUsageChunk(body []byte) bool {
	document := decode(body)
	if document == nil {
		return true
	}
	options, ok := document["stream_options"].(map[string]any)
	if !ok {
		return false
	}
	include, _ := options["include_usage"].(bool)
	return include
}

func keepUsageChunk(request *http.Request) bool {
	keep, recorded := request.Context().Value(usageIntentKey{}).(bool)
	return !recorded || keep
}

func metered(request *http.Request) bool {
	return request != nil && strings.EqualFold(request.Header.Get(usageRequestHeader), "true")
}

func modifyResponse(response *http.Response) error {
	if !isEventStreamContentType(response.Header.Get("Content-Type")) {
		if metered(response.Request) {
			return annotateUsage(response)
		}
		return nil
	}

	response.Header.Del("Content-Length")
	response.ContentLength = -1
	var trailer http.Header
	if metered(response.Request) {
		trailer = http.Header{usageResponseHeader: nil}
		response.Trailer = trailer
	}
	response.Body = newStreamFilter(response.Body, trailer, keepUsageChunk(response.Request))
	return nil
}

func annotateUsage(response *http.Response) error {
	var body bytes.Buffer
	if _, err := body.ReadFrom(io.LimitReader(response.Body, maxBufferBytes+1)); err != nil {
		return err
	}

	if body.Len() > maxBufferBytes {
		response.Body = &requestBody{
			Reader: io.MultiReader(bytes.NewReader(body.Bytes()), response.Body),
			Closer: response.Body,
		}
		return nil
	}

	if err := response.Body.Close(); err != nil {
		return err
	}
	if usage := extractUsage(decode(body.Bytes())); len(usage) > 0 {
		response.Header.Set(usageResponseHeader, formatUsage(usage))
	}
	response.Body = io.NopCloser(bytes.NewReader(body.Bytes()))
	return nil
}

type closeOnceReadCloser struct {
	io.ReadCloser
	once sync.Once
	err  error
}

func (c *closeOnceReadCloser) Close() error {
	c.once.Do(func() {
		c.err = c.ReadCloser.Close()
	})
	return c.err
}

type streamResponseBody struct {
	*io.PipeReader
	upstream io.Closer
}

func (b *streamResponseBody) Close() error {
	return errors.Join(b.PipeReader.Close(), b.upstream.Close())
}

func newStreamFilter(body io.ReadCloser, trailer http.Header, keep bool) io.ReadCloser {
	upstream := &closeOnceReadCloser{ReadCloser: body}
	reader, writer := io.Pipe()

	go func() {
		defer upstream.Close()

		var usage map[string]any
		var event bytes.Buffer
		scanner := bufio.NewScanner(upstream)
		scanner.Buffer(make([]byte, 0, initialEventBytes), maxEventBytes)
		scanner.Split(splitEvents)

		for scanner.Scan() {
			event.Reset()
			counts, dropped := rewriteEvent(scanner.Bytes(), keep, &event)
			if len(counts) > 0 {
				usage = counts
			}
			if dropped {
				continue
			}
			if _, err := writer.Write(event.Bytes()); err != nil {
				writer.CloseWithError(err)
				return
			}
		}
		if err := scanner.Err(); err != nil {
			writer.CloseWithError(err)
			return
		}

		if trailer != nil && len(usage) > 0 {
			trailer[usageResponseHeader] = []string{formatUsage(usage)}
		}
		writer.Close()
	}()

	return &streamResponseBody{PipeReader: reader, upstream: upstream}
}

func splitEvents(data []byte, atEOF bool) (int, []byte, error) {
	if index := bytes.Index(data, eventSeparator); index >= 0 {
		return index + len(eventSeparator), data[:index], nil
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

func rewriteEvent(event []byte, keep bool, out *bytes.Buffer) (map[string]any, bool) {
	lines := bytes.Split(event, newline)
	for index, line := range lines {
		if !bytes.HasPrefix(line, dataPrefix) {
			continue
		}

		document := decode(bytes.TrimSpace(line[len(dataPrefix):]))
		if document == nil {
			break
		}
		usage := extractUsage(document)
		if isCountsOnly(document) && !keep {
			return usage, true
		}
		if err := addPadding(document); err != nil {
			log.Printf("cannot pad chunk: %v", err)
			break
		}
		encoded, err := json.Marshal(document)
		if err != nil {
			log.Printf("cannot re-encode chunk: %v", err)
			break
		}

		lines[index] = append(append([]byte{}, dataLinePrefix...), encoded...)
		out.Write(bytes.Join(lines, newline))
		out.Write(eventSeparator)
		return usage, false
	}

	out.Write(event)
	out.Write(eventSeparator)
	return nil, false
}

func decode(data []byte) map[string]any {
	var document map[string]any
	if json.Unmarshal(data, &document) != nil {
		return nil
	}
	return document
}

func isEventStreamContentType(contentType string) bool {
	mediaType, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return false
	}
	return mediaType == "text/event-stream"
}

func addPadding(document map[string]any) error {
	choices, ok := document["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	first, ok := choices[0].(map[string]any)
	if !ok {
		return nil
	}
	delta, ok := first["delta"].(map[string]any)
	if !ok {
		return nil
	}

	span := big.NewInt(int64(len(paddingCharset) - minPaddingLength + 1))
	length, err := rand.Int(rand.Reader, span)
	if err != nil {
		return err
	}
	delta["p"] = paddingCharset[:minPaddingLength+int(length.Int64())]
	return nil
}

func isCountsOnly(document map[string]any) bool {
	choices, ok := document["choices"].([]any)
	if !ok || len(choices) != 0 {
		return false
	}
	_, ok = document["usage"].(map[string]any)
	return ok
}

func extractUsage(document map[string]any) map[string]any {
	if nested, ok := document["response"].(map[string]any); ok {
		if usage, ok := nested["usage"].(map[string]any); ok {
			return usage
		}
	}
	if usage, ok := document["usage"].(map[string]any); ok {
		return usage
	}
	return nil
}

func formatUsage(usage map[string]any) string {
	prompt := count(usage, "prompt_tokens", "input_tokens")
	completion := count(usage, "completion_tokens", "output_tokens")
	total := count(usage, "total_tokens")
	if total == 0 {
		total = prompt + completion
	}

	fields := []string{
		fmt.Sprintf("prompt=%d", prompt),
		fmt.Sprintf("completion=%d", completion),
		fmt.Sprintf("total=%d", total),
	}

	details, ok := usage["prompt_tokens_details"].(map[string]any)
	if !ok {
		details, ok = usage["input_tokens_details"].(map[string]any)
	}
	if ok {
		cached := min(prompt, count(details, "cached_tokens"))
		fields = append(fields,
			fmt.Sprintf("cached_prompt_tokens=%d", cached),
			fmt.Sprintf("uncached_prompt_tokens=%d", prompt-cached),
		)
	}
	return strings.Join(fields, ",")
}

func count(source map[string]any, names ...string) int {
	for _, name := range names {
		value, ok := source[name].(float64)
		if ok && value >= 0 && value == float64(int(value)) {
			return int(value)
		}
	}
	return 0
}
