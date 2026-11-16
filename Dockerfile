FROM golang:1.26.6-alpine@sha256:af8d6740070b8906d12eae1c3e3ea0957fb63f492051ea05e354c38ef9fe88df AS build
WORKDIR /src
COPY go.mod *.go ./
ENV GOTOOLCHAIN=local
RUN CGO_ENABLED=0 go build -mod=readonly -trimpath -buildvcs=false \
    -ldflags="-s -w -buildid=" -o /inference-sidecar .

FROM scratch
COPY --from=build /inference-sidecar /inference-sidecar
