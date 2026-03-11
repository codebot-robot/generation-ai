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
	"bufio"
	"context"
	"crypto/tls"
	"flag"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	server := flag.String("server", "http://bema.default.svc.cluster.local", "The server address (e.g., http://localhost:50051 or https://bema.example.com)")
	sessionID := flag.String("session", "", "The session ID to resume")
	list := flag.Bool("list", false, "List all sessions")
	flag.Parse()

	addr, dialOpts, cleanup, err := parseServer(*server)
	if err != nil {
		klog.Fatalf("failed to parse server: %v", err)
	}
	defer cleanup()

	conn, err := grpc.NewClient(addr, dialOpts...)
	if err != nil {
		klog.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	client := pb.NewBemaServiceClient(conn)

	if *list {
		listSessions(client)
		return
	}

	ctx := context.Background()
	var session *pb.Session
	if *sessionID == "" {
		session, err = client.CreateSession(ctx, &pb.CreateSessionRequest{})
		if err != nil {
			klog.Fatalf("failed to create session: %v", err)
		}
		fmt.Printf("Created new session: %s\n", session.Id)
	} else {
		session, err = client.GetSession(ctx, &pb.GetSessionRequest{Id: *sessionID})
		if err != nil {
			klog.Fatalf("failed to get session: %v", err)
		}
		fmt.Printf("Resuming session: %s\n", session.Id)
		for _, msg := range session.Messages {
			printMessage(msg)
		}
	}

	runChat(client, session.Id)
}

func parseServer(server string) (string, []grpc.DialOption, func(), error) {
	u, err := url.Parse(server)
	if err != nil {
		return "", nil, nil, fmt.Errorf("failed to parse server URL: %v", err)
	}

	var dialOpts []grpc.DialOption
	addr := u.Host

	switch u.Scheme {
	case "https":
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{})))
	case "http":
		dialOpts = append(dialOpts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	default:
		return "", nil, nil, fmt.Errorf("unknown scheme: %s", u.Scheme)
	}

	if strings.HasSuffix(u.Hostname(), ".svc.cluster.local") {
		// service.namespace.svc.cluster.local
		parts := strings.Split(u.Hostname(), ".")
		if len(parts) < 2 {
			return "", nil, nil, fmt.Errorf("invalid kubernetes service URL: %s", u.Hostname())
		}
		serviceName := parts[0]
		namespace := parts[1]

		remotePort := u.Port()
		if remotePort == "" {
			remotePort = "50051" // Default Bema port
		}

		// Find a free local port
		l, err := net.Listen("tcp", "localhost:0")
		if err != nil {
			return "", nil, nil, fmt.Errorf("failed to find free local port: %v", err)
		}
		localAddr := l.Addr().String()
		l.Close()
		_, localPort, _ := net.SplitHostPort(localAddr)

		klog.Infof("starting kubectl port-forward to %s in namespace %s", serviceName, namespace)
		cmd := exec.Command("kubectl", "port-forward", "-n", namespace, "svc/"+serviceName, localPort+":"+remotePort)
		if err := cmd.Start(); err != nil {
			return "", nil, nil, fmt.Errorf("failed to start kubectl port-forward: %v", err)
		}

		cleanup := func() {
			if cmd.Process != nil {
				cmd.Process.Kill()
			}
		}

		// Try to connect to the local port until it's ready
		ready := false
		for i := 0; i < 20; i++ {
			conn, err := net.DialTimeout("tcp", "localhost:"+localPort, 100*time.Millisecond)
			if err == nil {
				conn.Close()
				ready = true
				break
			}
			time.Sleep(200 * time.Millisecond)
		}

		if !ready {
			cleanup()
			return "", nil, nil, fmt.Errorf("kubectl port-forward failed to become ready")
		}

		addr = "localhost:" + localPort
		return addr, dialOpts, cleanup, nil
	}

	return addr, dialOpts, func() {}, nil
}

func listSessions(client pb.BemaServiceClient) {
	resp, err := client.ListSessions(context.Background(), &pb.ListSessionsRequest{})
	if err != nil {
		klog.Fatalf("failed to list sessions: %v", err)
	}
	fmt.Println("Sessions:")
	for _, s := range resp.Sessions {
		updated := "never"
		if s.UpdatedAt != nil {
			updated = s.UpdatedAt.AsTime().Format("2006-01-02 15:04:05")
		}
		fmt.Printf("- %s (updated: %s)\n", s.Id, updated)
	}
}

func printMessage(msg *pb.Message) {
	role := strings.ToUpper(msg.Role)
	var sb strings.Builder
	for _, p := range msg.Parts {
		switch part := p.Data.(type) {
		case *pb.Part_Text:
			sb.WriteString(part.Text)
		case *pb.Part_FunctionCall:
			sb.WriteString(fmt.Sprintf("[TOOL CALL: %s args: %v]", part.FunctionCall.Name, part.FunctionCall.Args.AsMap()))
		case *pb.Part_FunctionResponse:
			sb.WriteString(fmt.Sprintf("[TOOL RESPONSE: %s response: %v]", part.FunctionResponse.Name, part.FunctionResponse.Response.AsMap()))
		}
	}
	fmt.Printf("[%s]: %s\n", role, sb.String())
}

func runChat(client pb.BemaServiceClient, sessionID string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Watch for new messages
	stream, err := client.WatchSession(ctx, &pb.WatchSessionRequest{Id: sessionID})
	if err != nil {
		klog.Fatalf("failed to watch session: %v", err)
	}

	// We've already printed history in main if resuming.
	// WatchSession sends TYPE_UPDATED with full session first.
	// We need to skip messages we've already seen or just skip the first event if it's TYPE_UPDATED.
	firstEvent := true

	go func() {
		for {
			event, err := stream.Recv()
			if err == io.EOF {
				return
			}
			if err != nil {
				if ctx.Err() == nil {
					klog.Errorf("watch error: %v", err)
				}
				return
			}

			if firstEvent {
				firstEvent = false
				if event.Type == pb.SessionEvent_UPDATED {
					continue
				}
			}

			if event.Type == pb.SessionEvent_MESSAGE_APPENDED {
				msgs := event.Session.Messages
				if len(msgs) > 0 {
					lastMsg := msgs[len(msgs)-1]
					if lastMsg.Role != "user" {
						fmt.Print("\r\033[K") // Clear the prompt line
						printMessage(lastMsg)
						fmt.Print("> ")
					}
				}
			}
		}
	}()

	scanner := bufio.NewScanner(os.Stdin)
	fmt.Print("> ")
	for scanner.Scan() {
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			fmt.Print("> ")
			continue
		}
		if text == "/quit" || text == "/exit" {
			break
		}
		if strings.HasPrefix(text, "/session ") {
			newID := strings.TrimPrefix(text, "/session ")
			fmt.Printf("Switching to session: %s\n", newID)
			cancel() // Stop current watch
			runChat(client, newID)
			return
		}

		_, err := client.AppendMessage(ctx, &pb.AppendMessageRequest{
			Id: sessionID,
			Message: &pb.Message{
				Role: "user",
				Parts: []*pb.Part{
					{
						Data: &pb.Part_Text{
							Text: text,
						},
					},
				},
			},
		})
		if err != nil {
			klog.Errorf("failed to append message: %v", err)
		}
		// Do not print prompt here, let the watch goroutine or next loop handle it.
		// Actually, we need to print it if we want to wait for more user input.
		fmt.Print("> ")
	}
}
