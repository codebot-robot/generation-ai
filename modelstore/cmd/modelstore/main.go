// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"

	"github.com/gke-labs/generation-ai/modelstore/pkg/proxy"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	defer klog.Flush()

	if err := run(context.Background()); err != nil {
		klog.ErrorS(err, "terminated with error")
		os.Exit(1)
	}
}

func run(ctx context.Context) error {
	port := flag.String("port", "8080", "Port to listen on")
	cacheDir := flag.String("cache-dir", "/cache", "Directory to store cached models")
	upstream := flag.String("upstream", "https://huggingface.co", "Upstream URL to proxy")
	flag.Parse()

	if err := os.MkdirAll(*cacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	p, err := proxy.NewProxy(*upstream, *cacheDir)
	if err != nil {
		return fmt.Errorf("failed to create proxy: %w", err)
	}

	klog.InfoS("Starting modelstore", "port", *port, "cacheDir", *cacheDir, "upstream", *upstream)

	server := &http.Server{
		Addr:    ":" + *port,
		Handler: p,
	}

	return server.ListenAndServe()
}
