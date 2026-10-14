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

	"github.com/gke-labs/generation-ai/modelstore/apis/v1alpha1"
	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/gke-labs/generation-ai/modelstore/pkg/importer"
	"github.com/gke-labs/generation-ai/modelstore/pkg/proxy"
	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
)

type ServeOptions struct {
	Port string
	// BlobStore is where blobs live: a gs://bucket[/prefix] URL or a
	// local directory.
	BlobStore string
	// HFEndpoint is the Hugging Face Hub to import from.
	HFEndpoint string
}

func (o *ServeOptions) InitDefaults() {
	o.Port = "8080"
	o.BlobStore = "/cache"
	o.HFEndpoint = "https://huggingface.co"
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
	cmd.Flags().StringVar(&opt.BlobStore, "blob-store", opt.BlobStore,
		"Where blobs are stored: a gs://bucket[/prefix] URL (application default credentials) or a local directory")
	cmd.Flags().StringVar(&opt.BlobStore, "cache-dir", opt.BlobStore, "Deprecated alias for --blob-store")
	cmd.Flags().StringVar(&opt.HFEndpoint, "hf-endpoint", opt.HFEndpoint,
		"Hugging Face Hub to import from (HF_TOKEN in the environment is sent for gated repositories)")
	if err := cmd.Flags().MarkDeprecated("cache-dir", "use --blob-store"); err != nil {
		panic(err) // a typo in the flag name above, caught at startup
	}

	return cmd
}

func RunServe(ctx context.Context, opt *ServeOptions) error {
	log := klog.FromContext(ctx)
	store, err := blobstore.Open(ctx, opt.BlobStore)
	if err != nil {
		return fmt.Errorf("opening blob store %s: %w", opt.BlobStore, err)
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

	p := proxy.NewProxy(store, importer.New(store, opt.HFEndpoint), kube)
	if err := p.Resume(ctx); err != nil {
		return fmt.Errorf("resuming interrupted imports: %w", err)
	}

	log.Info("Starting modelstore", "port", opt.Port, "blobStore", store.String(), "hfEndpoint", opt.HFEndpoint)

	server := &http.Server{
		Addr:    ":" + opt.Port,
		Handler: p,
	}

	return server.ListenAndServe()
}
