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
	"errors"
	"fmt"
	"net/http"
	_ "net/http/pprof"
	"os"
	"time"

	"github.com/gke-labs/generation-ai/modelstore/pkg/blobstore"
	"github.com/gke-labs/generation-ai/modelstore/pkg/importer"
	"github.com/gke-labs/generation-ai/modelstore/pkg/proxy"
	"github.com/gke-labs/generation-ai/modelstore/pkg/registry"
	"github.com/spf13/cobra"
	"k8s.io/klog/v2"
)

type ServeOptions struct {
	Port string
	// BlobStore is where blobs live: a gs://bucket[/prefix] URL or a
	// local directory.
	BlobStore string
	// HFEndpoint is the Hugging Face Hub to import from.
	HFEndpoint string
	// PprofAddr is the address to serve net/http/pprof on (off by default).
	PprofAddr string
	// SignerServiceAccount is the Google service account to impersonate
	// for signing direct blob URLs (empty means no direct URLs).
	SignerServiceAccount string
}

func (o *ServeOptions) InitDefaults() {
	o.Port = "8080"
	o.BlobStore = "/cache"
	o.HFEndpoint = "https://huggingface.co"
	o.PprofAddr = ""
	o.SignerServiceAccount = ""
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
	cmd.Flags().StringVar(&opt.PprofAddr, "pprof-addr", opt.PprofAddr,
		"Address to serve net/http/pprof on (e.g. localhost:6060; disabled by default)")
	cmd.Flags().StringVar(&opt.SignerServiceAccount, "signer-service-account", opt.SignerServiceAccount,
		"Google service account to impersonate for signing direct blob URLs (empty means no direct URLs)")
	cmd.Flags().StringVar(&opt.SignerServiceAccount, "blob-signer-account", opt.SignerServiceAccount,
		"Alias for --signer-service-account")
	if err := cmd.Flags().MarkDeprecated("cache-dir", "use --blob-store"); err != nil {
		panic(err) // a typo in the flag name above, caught at startup
	}

	return cmd
}

func RunServe(ctx context.Context, opt *ServeOptions) error {
	log := klog.FromContext(ctx)
	if opt.SignerServiceAccount == "" {
		if env := os.Getenv("SIGNER_SERVICE_ACCOUNT"); env != "" {
			opt.SignerServiceAccount = env
		} else if env := os.Getenv("BLOB_SIGNER_ACCOUNT"); env != "" {
			opt.SignerServiceAccount = env
		}
	}
	var openOpts []blobstore.Option
	if opt.SignerServiceAccount != "" {
		openOpts = append(openOpts, blobstore.WithSignerServiceAccount(opt.SignerServiceAccount))
	}
	store, err := blobstore.Open(ctx, opt.BlobStore, openOpts...)
	if err != nil {
		return fmt.Errorf("opening blob store %s: %w", opt.BlobStore, err)
	}
	reg := registry.New(store)
	p := proxy.NewProxy(reg, importer.New(store, opt.HFEndpoint))

	log.Info("Starting modelstore", "port", opt.Port, "blobStore", store.String(), "hfEndpoint", opt.HFEndpoint, "signerServiceAccount", opt.SignerServiceAccount)

	if opt.PprofAddr != "" {
		pprofServer := &http.Server{
			Addr:              opt.PprofAddr,
			Handler:           http.DefaultServeMux,
			ReadHeaderTimeout: 10 * time.Second,
		}
		go func() {
			log.Info("Starting pprof server", "addr", opt.PprofAddr)
			if err := pprofServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Error(err, "pprof server failed", "addr", opt.PprofAddr)
			}
		}()
		defer func() {
			if err := pprofServer.Close(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Error(err, "closing pprof server", "addr", opt.PprofAddr)
			}
		}()
	}

	server := &http.Server{
		Addr:              ":" + opt.Port,
		Handler:           p,
		ReadHeaderTimeout: 30 * time.Second,
	}
	return server.ListenAndServe()
}
