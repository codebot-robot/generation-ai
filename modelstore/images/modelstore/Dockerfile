# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM golang:1.23-alpine AS builder

WORKDIR /app
COPY go.mod ./
# No go.sum yet, will be generated if needed
RUN go mod download

COPY . .
RUN go build -o modelstore ./cmd/modelstore

FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=builder /app/modelstore .
RUN mkdir /cache
EXPOSE 8080
ENTRYPOINT ["./modelstore"]
CMD ["--port=8080", "--cache-dir=/cache"]
