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
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

// Models by name (#357): `vxpu register` makes a model servable by name
// (the router imports its weights into modelstore and compiles it), and
// `vxpu ask --model` opens a session on it by name.

// connectRouter dials the router at addr, or ensures the router pod
// exists (creating it with settings if needed) and port-forwards to it.
// done closes the connection and the tunnel.
func connectRouter(ctx context.Context, addr, pod string, settings routerSettings) (pb.ExecutorClient, func(), error) {
	log := klog.FromContext(ctx)
	stop := func() {}
	if addr == "" {
		// A running router is used as it is: its settings were decided
		// by `vxpu up`; these only apply when the router has to be created.
		if err := ensurePod(ctx, pod, settings, false); err != nil {
			return nil, nil, fmt.Errorf("ensuring router pod: %w", err)
		}
		var err error
		if addr, stop, err = portForward(ctx, pod); err != nil {
			return nil, nil, fmt.Errorf("port-forward: %w", err)
		}
	}
	conn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallSendMsgSize(maxMessageBytes),
			grpc.MaxCallRecvMsgSize(maxMessageBytes)))
	if err != nil {
		stop()
		return nil, nil, fmt.Errorf("dialing %s: %w", addr, err)
	}
	return pb.NewExecutorClient(conn), func() {
		if err := conn.Close(); err != nil {
			log.Error(err, "closing router connection", "address", addr)
		}
		stop()
	}, nil
}

// splitRevision splits REPO[@REVISION].
func splitRevision(ref string) (string, string) {
	if repo, revision, ok := strings.Cut(ref, "@"); ok {
		return repo, revision
	}
	return ref, ""
}

func cmdRegister(ctx context.Context, args []string) error {
	flags := flag.NewFlagSet("register", flag.ContinueOnError)
	pod := flags.String("pod", "vxpu-router", "router pod name")
	routerAddr := flags.String("router", "", "vxpu-router address (e.g. localhost:50051); if set, skips the pod and port-forward")
	timeout := flags.Duration("timeout", 2*time.Hour, "how long to wait (importing weights and compiling can take a while)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return fmt.Errorf("usage: vxpu register [flags] REPO[@REVISION]")
	}
	repo, revision := splitRevision(flags.Arg(0))
	client, done, err := connectRouter(ctx, *routerAddr, *pod, routerSettings{})
	if err != nil {
		return err
	}
	defer done()
	ctx, cancel := context.WithTimeout(ctx, *timeout)
	defer cancel()
	fmt.Printf("registering %s (importing weights into modelstore, compiling if needed)...\n", flags.Arg(0))
	started := time.Now()
	resp, err := client.RegisterModel(ctx, &pb.RegisterModelRequest{Model: repo, Revision: revision}, grpc.WaitForReady(true))
	if err != nil {
		return fmt.Errorf("RegisterModel: %w", err)
	}
	how := "already compiled for this engine"
	if resp.Compiled {
		how = "compiled"
	}
	fmt.Printf("registered %s@%s (commit %s): %s, %.1f GB of weights, artifact %s (%.0fs)\n",
		resp.Model, resp.Revision, resp.Commit, how, float64(resp.WeightBytes)/1e9, resp.Artifact, time.Since(started).Seconds())
	return nil
}

// askByName opens a session on a registered model by name (the router
// loads it the first time; NewSession fails with FAILED_PRECONDITION
// until it is loaded) and runs one chat turn.
func askByName(ctx context.Context, client pb.ExecutorClient, repo, revision, prompt string, maxNewTokens int) error {
	started := time.Now()
	var session *pb.NewSessionResponse
	for {
		var err error
		session, err = client.NewSession(ctx, &pb.NewSessionRequest{Model: repo, Revision: revision}, grpc.WaitForReady(true))
		if err == nil {
			break
		}
		if status.Code(err) != codes.FailedPrecondition || strings.Contains(err.Error(), "not registered") ||
			strings.Contains(err.Error(), "not compiled") {
			return fmt.Errorf("NewSession: %w", err)
		}
		if ctx.Err() != nil {
			return fmt.Errorf("timed out waiting for %s to load: %w", repo, err)
		}
		fmt.Printf("  loading %s... (%.0fs)\n", repo, time.Since(started).Seconds())
		time.Sleep(15 * time.Second)
	}
	fmt.Printf("session on %s ready in %.1fs\n", repo, time.Since(started).Seconds())
	started = time.Now()
	reply, err := client.Chat(ctx, &pb.ChatRequest{SessionId: session.SessionId, Text: prompt, MaxNewTokens: int32(maxNewTokens)})
	if err != nil {
		return fmt.Errorf("Chat: %w", err)
	}
	fmt.Printf("\n>>> %s\n%s\n\n(%d tokens, %.1f ms/token on the executor, %.1fs round trip)\n",
		prompt, reply.Text, reply.Generated, reply.MsPerToken, time.Since(started).Seconds())
	return nil
}
