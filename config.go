package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"

	"github.com/tinfoilsh/inference-sidecar/internal/key"
	localjwt "github.com/tinfoilsh/inference-sidecar/internal/key/jwt"
	"github.com/tinfoilsh/inference-sidecar/internal/key/online"
	"golang.org/x/time/rate"
)

// Policy is supplied as SIDECAR_POLICY in the measured container environment.
// DOMAIN is explicitly passed through by the workload's existing env mechanism.
type Policy struct {
	Version                int       `json:"version"`
	Authenticated          *bool     `json:"authenticated"`
	ControlPlane           string    `json:"control-plane"`
	AuthenticatedEndpoints *[]string `json:"authenticated-endpoints"`
	Paths                  []string  `json:"paths"`
	OriginDomains          []string  `json:"origins"`
	RateLimit              float64   `json:"rate-limit"`
	RateBurst              int       `json:"rate-burst"`
	TrustedProxies         []string  `json:"trusted-proxies"`
}

func parsePolicy(raw, domain string) (*Policy, error) {
	var policy Policy
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&policy); err != nil {
		return nil, fmt.Errorf("SIDECAR_POLICY: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return nil, fmt.Errorf("SIDECAR_POLICY must contain one JSON object")
	}
	if policy.Version != 1 || policy.Authenticated == nil {
		return nil, fmt.Errorf("SIDECAR_POLICY requires version 1 and explicit authenticated")
	}
	if policy.RateLimit < 0 || policy.RateBurst < 0 || (policy.RateLimit > 0 && policy.RateBurst == 0) {
		return nil, fmt.Errorf("rate-limit and rate-burst must be nonnegative, with a positive burst when enabled")
	}
	for _, proxy := range policy.TrustedProxies {
		if net.ParseIP(proxy) == nil {
			return nil, fmt.Errorf("trusted proxy must be an exact IP: %q", proxy)
		}
	}
	if *policy.Authenticated {
		if domain == "" || strings.ContainsAny(domain, "/: ,\t\r\n") {
			return nil, fmt.Errorf("authenticated policy requires an explicit deployment DOMAIN")
		}
		control, err := url.Parse(policy.ControlPlane)
		if err != nil || control.Scheme != "https" || control.Host == "" || control.User != nil || control.RawQuery != "" || control.Fragment != "" {
			return nil, fmt.Errorf("authenticated policy requires an HTTPS control-plane URL")
		}
	}
	return &policy, nil
}

func (p *Policy) validators() (key.Validator, *RateLimiter, error) {
	var validator key.Validator
	if *p.Authenticated {
		control, _ := url.Parse(p.ControlPlane) // validated before starting the child
		onlineValidator, err := online.NewValidator(control.JoinPath("api", "shim", "validate-key").String())
		if err != nil {
			return nil, nil, err
		}
		jwtValidator := localjwt.NewValidator(control.JoinPath(".well-known", "jwks.json").String(), p.ControlPlane, localjwt.AccessTokenAudience, localjwt.RequiredScope)
		validator = &metricsValidator{online: onlineValidator, chain: key.NewChain(jwtValidator, onlineValidator)}
	}
	var limiter *RateLimiter
	if p.RateLimit > 0 {
		limiter = NewRateLimiter(rate.Limit(p.RateLimit), p.RateBurst)
	}
	return validator, limiter, nil
}
