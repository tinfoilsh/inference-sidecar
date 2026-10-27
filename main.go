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
	"time"
)

const (
	usageRequestHeader  = "X-Tinfoil-Request-Usage-Metrics"
	usageResponseHeader = "X-Tinfoil-Usage-Metrics"
	upstreamHost        = "127.0.0.1"
	defaultUpstreamPort = "18001"
	defaultListenPort   = "8000"
	maxBufferBytes      = 8 << 20
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
	os.Exit(run())
}

func run() int {
	if len(os.Args) == 2 && os.Args[1] == "--check-policy-version=1" {
		fmt.Println("1")
		return 0
	}
	policy, err := parsePolicy(os.Getenv("SIDECAR_POLICY"), os.Getenv("DOMAIN"))
	if err != nil {
		log.Print(err)
		return 1
	}
	child := os.Args[1:]
	// A measured entrypoint uses this guard so a pre-policy sidecar fails to
	// start instead of silently ignoring the new environment configuration.
	if len(child) > 0 && strings.HasPrefix(child[0], "--policy-version=") {
		if child[0] != "--policy-version=1" {
			log.Print("unsupported policy version")
			return 1
		}
		child = child[1:]
	}
	upstreamPort := env("SIDECAR_UPSTREAM_PORT", defaultUpstreamPort)
	listenPort := env("SIDECAR_LISTEN", lastPortFlag(child, defaultListenPort))
	upstream := &url.URL{Scheme: "http", Host: net.JoinHostPort(upstreamHost, upstreamPort)}
	if explicit := os.Getenv("SIDECAR_UPSTREAM_URL"); explicit != "" {
		upstream, err = url.Parse(explicit)
		if err != nil || upstream.Scheme != "http" || upstream.User != nil || upstream.RawQuery != "" || upstream.Fragment != "" || (upstream.Path != "" && upstream.Path != "/") || !net.ParseIP(upstream.Hostname()).IsLoopback() || upstream.Port() == "" {
			log.Print("SIDECAR_UPSTREAM_URL must be an HTTP loopback address with an explicit port")
			return 1
		}
	} else {
		if len(child) == 0 {
			log.Print("no upstream command or SIDECAR_UPSTREAM_URL given")
			return 1
		}
		child = upstreamCommand(child, upstreamPort)
	}
	validator, limiter, err := policy.validators()
	if err != nil {
		log.Print(err)
		return 1
	}
	listener, err := net.Listen("tcp", net.JoinHostPort("", listenPort))
	if err != nil {
		log.Printf("cannot listen: %v", err)
		return 1
	}
	defer listener.Close()
	handler := withPolicy(policy, os.Getenv("DOMAIN"), validator, limiter, newHandler(upstream, os.Getenv("SIDECAR_UPSTREAM_URL") == ""))
	var active sync.WaitGroup
	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			active.Add(1)
			defer active.Done()
			handler.ServeHTTP(w, r)
		}),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       2 * time.Minute,
	}
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(signals)
	var cmd *exec.Cmd
	childDone := make(chan error, 1)
	if len(child) > 0 {
		cmd = exec.Command(child[0], child[1:]...)
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
		if err := cmd.Start(); err != nil {
			log.Printf("cannot start %s: %v", child[0], err)
			return 1
		}
		go func() { childDone <- cmd.Wait() }()
	}
	served := make(chan error, 1)
	go func() { served <- server.Serve(listener) }()
	select {
	case err := <-childDone:
		server.Close()
		if cmd.ProcessState == nil {
			log.Printf("cannot wait for child: %v", err)
			return 1
		}
		return exitCode(cmd.ProcessState)
	case received := <-signals:
		// Keep the existing upstream alive until in-flight responses drain.
		_ = server.Shutdown(context.Background())
		active.Wait()
		if cmd != nil {
			_ = cmd.Process.Signal(received)
			<-childDone
			return exitCode(cmd.ProcessState)
		}
		return 0
	case err := <-served:
		log.Printf("listener stopped: %v", err)
		if cmd != nil {
			_ = cmd.Process.Signal(syscall.SIGTERM)
			<-childDone
		}
		return 1
	}
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

// Explicit upstream mode is transparent: MCP, document uploads, routers and
// WebSockets retain their own response formats and body admission behavior.
func newHandler(upstream *url.URL, inference bool) http.Handler {
	proxy := &httputil.ReverseProxy{
		Rewrite: func(request *httputil.ProxyRequest) {
			request.SetURL(upstream)
			request.Out.Host = request.In.Host
			request.Out.Header.Set("X-Forwarded-Host", request.In.Host)
		},
		FlushInterval: -1,
		ModifyResponse: func(response *http.Response) error {
			response.Header.Del("Access-Control-Allow-Origin")
			if inference {
				return modifyResponse(response)
			}
			return nil
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			log.Printf("proxy error: %v", err)
			writeAPIError(w, errUpstreamUnreachable)
		},
	}
	if inference {
		return withUsageIntent(proxy)
	}
	return proxy
}

type usageIntentKey struct{}

type requestBody struct {
	io.Reader
	io.Closer
}

func withUsageIntent(next http.Handler) http.Handler {
	return http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method != http.MethodPost {
			next.ServeHTTP(writer, request)
			return
		}

		head, err := io.ReadAll(io.LimitReader(request.Body, maxBufferBytes))
		if err != nil {
			http.Error(writer, "cannot read request body", http.StatusBadRequest)
			return
		}
		request.Body = &requestBody{
			Reader: io.MultiReader(bytes.NewReader(head), request.Body),
			Closer: request.Body,
		}

		intent := context.WithValue(request.Context(), usageIntentKey{}, askedForUsageChunk(head))
		next.ServeHTTP(writer, request.WithContext(intent))
	})
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
