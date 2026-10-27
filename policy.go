package main

import (
	"errors"
	"log"
	"net"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"github.com/tinfoilsh/inference-sidecar/internal/key"
)

// pathMatchesPattern checks if a request path matches a pattern.
// Patterns can be exact matches or use a trailing * for segment-boundary prefix matching.
// Examples:
//   - "/v1/models" matches only "/v1/models"
//   - "/v1/user/*" matches "/v1/user/123", "/v1/user/abc/settings", etc.
func pathMatchesPattern(pattern, path string) bool {
	prefix, wildcard := strings.CutSuffix(pattern, "*")
	if !wildcard {
		return pattern == path
	}
	if strings.HasSuffix(prefix, "/") {
		return strings.HasPrefix(path, prefix)
	}
	return path == prefix || strings.HasPrefix(path, prefix+"/")
}

// pathAllowed checks if the request path matches any of the allowed patterns.
func pathAllowed(allowedPaths []string, path string) bool {
	for _, pattern := range allowedPaths {
		if pathMatchesPattern(pattern, path) {
			return true
		}
	}
	return false
}

func requestedHost(r *http.Request) string {
	host, _, err := net.SplitHostPort(r.Host)
	if err != nil {
		host = r.Host
	}
	return strings.ToLower(host)
}

// requiresAuth reports whether path requires API key authentication.
// If authenticatedEndpoints is nil (not configured), it defaults to only
// requiring auth for /v1/chat/completions for backwards compatibility.
// If authenticatedEndpoints is an empty slice, no paths require auth.
func requiresAuth(authenticatedEndpoints *[]string, path string) bool {
	if authenticatedEndpoints == nil {
		return path == "/v1/chat/completions"
	}
	return pathAllowed(*authenticatedEndpoints, path)
}

// metricsValidator validates /metrics online-only: the local JWT leg accepts
// any inference-scoped token without consulting the control plane, which would
// bypass its admin-key requirement for /metrics. All other paths use the full
// chain.
type metricsValidator struct {
	online key.Validator
	chain  key.Validator
}

func (v *metricsValidator) Validate(req key.Request) error {
	if req.Path == "/metrics" {
		return v.online.Validate(req)
	}
	return v.chain.Validate(req)
}

// extractBearerToken returns the token portion of an Authorization header,
// accepting any capitalization of the "Bearer" scheme.
func extractBearerToken(header string) string {
	const scheme = "bearer "
	if len(header) < len(scheme) || !strings.EqualFold(header[:len(scheme)], scheme) {
		return ""
	}
	return strings.TrimSpace(header[len(scheme):])
}

// writeValidationFailure maps a credential validation failure onto the
// client-facing error for its status. A 403 is a valid credential that lacks
// permission, which is distinct from a 401's bad or unknown credential.
func writeValidationFailure(w http.ResponseWriter, err error) {
	var validationErr *key.ValidationError
	if !errors.As(err, &validationErr) {
		writeAPIError(w, errServer)
		return
	}
	if validationErr.StatusCode < http.StatusBadRequest || validationErr.StatusCode > maxHTTPErrorStatus {
		writeAPIError(w, errServer)
		return
	}

	// Preserve 402 for exhausted credit so relays can distinguish it from
	// rate limiting without decrypting the error body.
	apiErr := errServer
	switch validationErr.StatusCode {
	case http.StatusUnauthorized:
		apiErr = errInvalidAPIKey
	case http.StatusForbidden:
		apiErr = errInsufficientPermissions
	case http.StatusPaymentRequired:
		apiErr = errQuotaExceeded
	case http.StatusTooManyRequests:
		apiErr = errRateLimited
		if validationErr.QuotaExceeded {
			apiErr = errQuotaExceeded
			apiErr.message = errMsgKeyQuotaExceeded
		}
	default:
		apiErr.message = http.StatusText(validationErr.StatusCode)
		if apiErr.message == "" {
			apiErr.message = errMsgValidationFailed
		}
		if validationErr.StatusCode < http.StatusInternalServerError {
			apiErr.errType = errTypeInvalidRequest
		} else if validationErr.StatusCode == http.StatusServiceUnavailable {
			apiErr.errType = errTypeServiceUnavailable
		}
	}
	apiErr.status = validationErr.StatusCode
	if validationErr.RetryAfter != "" {
		w.Header().Set("Retry-After", validationErr.RetryAfter)
	}
	writeAPIError(w, apiErr)
}

func corsMiddleware(config *Policy, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" {
			// Allow only configured origins
			if len(config.OriginDomains) > 0 && !slices.Contains(config.OriginDomains, origin) {
				// CORS origin not allowed
				writeAPIError(w, errOriginNotAllowed)
				return
			}

			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin") // cache
			w.Header().Set("Access-Control-Allow-Credentials", "true")
			w.Header().Set("Access-Control-Allow-Methods", "GET, PUT, POST, HEAD, DELETE, OPTIONS")
			w.Header().Set("Access-Control-Expose-Headers", "Ehbp-Encapsulated-Key, Ehbp-Response-Nonce, Content-Type, Tinfoil-Pt, Retry-After")

			// Echo requested headers or use a safe default
			reqHdr := r.Header.Get("Access-Control-Request-Headers")
			if reqHdr == "" {
				reqHdr = "Authorization, Content-Type, Ehbp-Encapsulated-Key"
			}
			w.Header().Set("Access-Control-Allow-Headers", reqHdr)

			if r.Method == http.MethodOptions {
				// CORS preflight
				w.WriteHeader(http.StatusNoContent)
				return
			}

			// CORS allowed
		}

		next.ServeHTTP(w, r)
	})
}

// withPolicy runs before request-body inspection and proxying, including on
// direct connections to the sidecar.
func withPolicy(config *Policy, domain string, validator key.Validator, rateLimiter *RateLimiter, next http.Handler) http.Handler {
	authorize := func(w http.ResponseWriter, r *http.Request) bool {
		apiKey := extractBearerToken(r.Header.Get("Authorization"))
		if validator != nil && (r.URL.Path == "/metrics" || requiresAuth(config.AuthenticatedEndpoints, r.URL.Path)) {
			if len(apiKey) == 0 {
				writeAPIError(w, errAPIKeyRequired)
				return false
			}

			validationReq := key.Request{
				APIKey:        apiKey,
				Domain:        strings.ToLower(domain),
				RequestedHost: requestedHost(r),
				Path:          r.URL.Path,
			}

			if err := validator.Validate(validationReq); err != nil {
				log.Printf("Warning: failed to validate API key: %v", err)
				writeValidationFailure(w, err)
				return false
			}
		}

		if rateLimiter != nil {
			if apiKey == "" {
				writeAPIError(w, errAPIKeyRequired)
				return false
			}
			limiter := rateLimiter.Limit(apiKey)
			if reservation := limiter.Reserve(); !reservation.OK() || reservation.Delay() > 0 {
				// Reserve reports how long until a token frees up; cancel so
				// the rejected request does not consume it.
				delay := reservation.Delay()
				reservation.Cancel()
				w.Header().Set("Retry-After", strconv.Itoa(retryAfterSeconds(delay)))
				writeAPIError(w, errRateLimited)
				return false
			}
		}
		return true
	}
	return corsMiddleware(config, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodConnect {
			http.Error(w, "CONNECT is not a workload HTTP endpoint", http.StatusMethodNotAllowed)
			return
		}
		if len(config.Paths) > 0 && !pathAllowed(config.Paths, r.URL.Path) {
			writeAPIError(w, errNotFound)
			return
		}
		// Only the measured shim bridge may supply the legacy rewritten Host.
		// All other clients, including direct callers, use their actual Host.
		if requestedHost(r) == "localhost" && trustedProxy(config.TrustedProxies, r.RemoteAddr) {
			hosts := r.Header.Values("X-Forwarded-Host")
			if len(hosts) != 1 || hosts[0] == "" || strings.ContainsAny(hosts[0], ", \t\r\n\"\\") {
				http.Error(w, "invalid original host", http.StatusBadRequest)
				return
			}
			r.Host = hosts[0]
		}
		// Routers consume X-Forwarded-Host. Never relay a caller's assertion.
		r.Header.Del("Forwarded")
		r.Header.Set("X-Forwarded-Host", r.Host)
		if authorize(w, r) {
			next.ServeHTTP(w, r)
		}
	}))
}

func trustedProxy(proxies []string, remote string) bool {
	host, _, err := net.SplitHostPort(remote)
	return err == nil && slices.Contains(proxies, host)
}
