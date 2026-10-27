package main

import (
	"encoding/json"
	"net/http"
	"time"
)

// Error type strings returned in API error responses, following OpenAI's
// taxonomy so client SDKs classify sidecar errors the same way they classify
// upstream ones. See https://platform.openai.com/docs/guides/error-codes
const (
	errTypeInvalidRequest     = "invalid_request_error"
	errTypeInsufficientQuota  = "insufficient_quota"
	errTypeRateLimit          = "rate_limit_error"
	errTypeServiceUnavailable = "service_unavailable_error"
	errTypeServer             = "server_error"
)

// Machine-readable error codes carried in the `code` field. Codes shared
// with OpenAI keep OpenAI's spelling.
const (
	errCodeMissingAPIKey           = "missing_api_key"
	errCodeInvalidAPIKey           = "invalid_api_key"
	errCodeInsufficientPermissions = "insufficient_permissions"
	errCodeInsufficientQuota       = "insufficient_quota"
	errCodeRateLimitExceeded       = "rate_limit_exceeded"
	errCodeOriginNotAllowed        = "origin_not_allowed"
	errCodeNotFound                = "not_found"
	errCodeUpstreamUnreachable     = "upstream_unreachable"
)

// Client-facing error messages, aligned with OpenAI's standard error messages
// where applicable.
const (
	errMsgAPIKeyRequired          = "You didn't provide an API key."
	errMsgInvalidAPIKey           = "Incorrect API key provided."
	errMsgInsufficientPermissions = "Your API key does not have permission to access this resource."
	errMsgQuotaExceeded           = "You exceeded your current quota, please check your plan and billing details."
	errMsgKeyQuotaExceeded        = "You exceeded this API key's usage or spending limit."
	errMsgValidationFailed        = "API key validation failed."
	errMsgRateLimited             = "Rate limit reached for requests."
	errMsgServerError             = "The server had an error while processing your request."
	errMsgUpstreamUnreachable     = "The inference service is temporarily unreachable. Please try again."
	errMsgOriginNotAllowed        = "CORS origin not allowed."
	errMsgNotFound                = "Not found."
)

const maxHTTPErrorStatus = 599

// retryAfterSeconds converts a rate limiter delay into a Retry-After value,
// rounding up so the client never retries inside the window it was just
// rejected in. Dividing before rounding keeps rate.InfDuration (MaxInt64)
// from overflowing into a misleadingly short hint.
func retryAfterSeconds(delay time.Duration) int {
	secs := int(delay / time.Second)
	if delay%time.Second != 0 {
		secs++
	}
	if secs < 1 {
		secs = 1
	}
	return secs
}

// apiError is an error response in OpenAI's format. param and code are
// emitted as JSON null when empty, matching OpenAI's envelope.
type apiError struct {
	status  int
	errType string
	code    string
	message string
}

var (
	errAPIKeyRequired = apiError{
		status:  http.StatusUnauthorized,
		errType: errTypeInvalidRequest,
		code:    errCodeMissingAPIKey,
		message: errMsgAPIKeyRequired,
	}
	errInvalidAPIKey = apiError{
		status:  http.StatusUnauthorized,
		errType: errTypeInvalidRequest,
		code:    errCodeInvalidAPIKey,
		message: errMsgInvalidAPIKey,
	}
	errInsufficientPermissions = apiError{
		status:  http.StatusForbidden,
		errType: errTypeInvalidRequest,
		code:    errCodeInsufficientPermissions,
		message: errMsgInsufficientPermissions,
	}
	errQuotaExceeded = apiError{
		status:  http.StatusPaymentRequired,
		errType: errTypeInsufficientQuota,
		code:    errCodeInsufficientQuota,
		message: errMsgQuotaExceeded,
	}
	errRateLimited = apiError{
		status:  http.StatusTooManyRequests,
		errType: errTypeRateLimit,
		code:    errCodeRateLimitExceeded,
		message: errMsgRateLimited,
	}
	errServer = apiError{
		status:  http.StatusInternalServerError,
		errType: errTypeServer,
		message: errMsgServerError,
	}
	errUpstreamUnreachable = apiError{
		status:  http.StatusBadGateway,
		errType: errTypeServer,
		code:    errCodeUpstreamUnreachable,
		message: errMsgUpstreamUnreachable,
	}
	errOriginNotAllowed = apiError{
		status:  http.StatusForbidden,
		errType: errTypeInvalidRequest,
		code:    errCodeOriginNotAllowed,
		message: errMsgOriginNotAllowed,
	}
	errNotFound = apiError{
		status:  http.StatusNotFound,
		errType: errTypeInvalidRequest,
		code:    errCodeNotFound,
		message: errMsgNotFound,
	}
)

// errorBody mirrors OpenAI's error object. Param and Code serialize as null
// when unset.
type errorBody struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}

type errorEnvelope struct {
	Error errorBody `json:"error"`
}

func nullableString(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

func (e apiError) envelope() errorEnvelope {
	return errorEnvelope{Error: errorBody{
		Message: e.message,
		Type:    e.errType,
		Code:    nullableString(e.code),
	}}
}

func writeAPIError(w http.ResponseWriter, e apiError) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(e.status)
	json.NewEncoder(w).Encode(e.envelope())
}
