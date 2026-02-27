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

package commands

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	"github.com/gke-labs/generation-ai/modelstore/pkg/proxy"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
)

type ServeOptions struct {
	Port     string
	CacheDir string
}

func (o *ServeOptions) InitDefaults() {
	o.Port = "8080"
	o.CacheDir = "/cache"
}

func BuildServeCommand() *cobra.Command {
	opt := &ServeOptions{}
	opt.InitDefaults()

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start the modelstore server",
		RunE: func(cmd *cobra.Command, args []string) error {
			return RunServe(cmd.Context(), opt)
		},
	}

	cmd.Flags().StringVar(&opt.Port, "port", opt.Port, "Port to listen on")
	cmd.Flags().StringVar(&opt.CacheDir, "cache-dir", opt.CacheDir, "Directory to store cached models")

	return cmd
}

func RunServe(ctx context.Context, opt *ServeOptions) error {
	log := klog.FromContext(ctx)
	if err := os.MkdirAll(opt.CacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory: %w", err)
	}

	scheme := runtime.NewScheme()
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		return fmt.Errorf("failed to add v1alpha1 to scheme: %w", err)
	}

	cfg, err := config.GetConfig()
	if err != nil {
		return fmt.Errorf("failed to get kubernetes config: %w", err)
	}

	kube, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		return fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	// For now we still use NewProxy but we will clean up its implementation
	p, err := proxy.NewProxy("", opt.CacheDir, kube)
	if err != nil {
		return fmt.Errorf("failed to create proxy: %w", err)
	}

	log.Info("Starting modelstore", "port", opt.Port, "cacheDir", opt.CacheDir)

	server := &http.Server{
		Addr:    ":" + opt.Port,
		Handler: p,
	}

	return server.ListenAndServe()
}
