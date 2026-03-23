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
	"net"
	"net/http"

	"github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	bemav1alpha1 "github.com/gke-labs/generation-ai/experiments/bema/pkg/apis/v1alpha1"
	"github.com/gke-labs/generation-ai/experiments/bema/pkg/backend/gemini"
	"github.com/gke-labs/generation-ai/experiments/bema/pkg/executors/sandbox"
	"github.com/gke-labs/generation-ai/experiments/bema/pkg/server"
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func main() {
	klog.InitFlags(nil)
	port := flag.String("port", "50051", "The server port")
	httpPort := flag.String("http-port", "8080", "The HTTP server port")
	tlsCertFile := flag.String("tls-cert-file", "", "File containing the TLS certificate")
	tlsKeyFile := flag.String("tls-key-file", "", "File containing the TLS private key")
	storageDir := flag.String("storage-dir", "/tmp/bema", "Directory to store sessions")
	backendType := flag.String("backend", "gemini", "The LLM backend to use (e.g. gemini)")
	modelName := flag.String("model", "gemini-3-flash-preview", "The model name for the backend")
	flag.Parse()

	ctx := context.Background()

	scheme := runtime.NewScheme()
	if err := bemav1alpha1.AddToScheme(scheme); err != nil {
		klog.Fatalf("failed to add bema v1alpha1 to scheme: %v", err)
	}

	config, err := ctrl.GetConfig()
	if err != nil {
		klog.Fatalf("failed to get kubeconfig: %v", err)
	}

	k8sClient, err := client.New(config, client.Options{Scheme: scheme})
	if err != nil {
		klog.Fatalf("failed to create k8s client: %v", err)
	}

	var backend server.Backend
	if *backendType == "gemini" {
		var err error
		backend, err = gemini.New(ctx, *modelName)
		if err != nil {
			klog.Fatalf("failed to create gemini backend: %v", err)
		}
		klog.Infof("using gemini backend with model %s", *modelName)
	}

	executor, err := sandbox.New(ctx)
	if err != nil {
		klog.Warningf("failed to create sandbox executor: %v", err)
	}

	lis, err := net.Listen("tcp", ":"+*port)
	if err != nil {
		klog.Fatalf("failed to listen: %v", err)
	}

	s := grpc.NewServer()
	store, err := server.NewFileSessionStore(*storageDir)
	if err != nil {
		klog.Fatalf("failed to create session store: %v", err)
	}

	bemaServer, err := server.NewBemaServer(store, backend, executor, k8sClient)
	if err != nil {
		klog.Fatalf("failed to create bema server: %v", err)
	}

	v1alpha1.RegisterBemaServiceServer(s, bemaServer)
	reflection.Register(s)

	apiServer := server.NewAPIServer(store)
	go func() {
		if *tlsCertFile != "" && *tlsKeyFile != "" {
			klog.Infof("HTTP server listening (TLS) at :%s", *httpPort)
			if err := http.ListenAndServeTLS(":"+*httpPort, *tlsCertFile, *tlsKeyFile, apiServer); err != nil {
				klog.Fatalf("failed to serve HTTPS: %v", err)
			}
		} else {
			klog.Infof("HTTP server listening at :%s", *httpPort)
			if err := http.ListenAndServe(":"+*httpPort, apiServer); err != nil {
				klog.Fatalf("failed to serve HTTP: %v", err)
			}
		}
	}()

	klog.Infof("server listening at %v", lis.Addr())
	if err := s.Serve(lis); err != nil {
		klog.Fatalf("failed to serve: %v", err)
	}
}
