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
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	addr := flag.String("addr", "localhost:50051", "The server address")
	sessionID := flag.String("session", "", "The session ID to resume")
	list := flag.Bool("list", false, "List all sessions")
	flag.Parse()

	conn, err := grpc.Dial(*addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
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
