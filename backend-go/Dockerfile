FROM golang:1.24-alpine AS builder

WORKDIR /src

COPY backend-go/go.mod ./backend-go/go.mod
RUN cd backend-go && go mod download

COPY backend-go ./backend-go
RUN cd backend-go && CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /out/api ./cmd/api

FROM alpine:3.20

WORKDIR /app

COPY --from=builder /out/api /app/api
COPY resources /app/resources

EXPOSE 9999

ENTRYPOINT ["/app/api"]
