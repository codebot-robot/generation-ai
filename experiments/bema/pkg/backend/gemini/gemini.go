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

package gemini

import (
	"context"
	"fmt"
	"os"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
	"k8s.io/klog/v2"
)

type GeminiBackend struct {
	client *genai.Client
	model  string
}

func New(ctx context.Context, model string) (*GeminiBackend, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	var opts []option.ClientOption
	if apiKey != "" {
		opts = append(opts, option.WithAPIKey(apiKey))
	}

	client, err := genai.NewClient(ctx, opts...)
	if err != nil {
		return nil, err
	}

	return &GeminiBackend{
		client: client,
		model:  model,
	}, nil
}

func (b *GeminiBackend) GenerateResponse(ctx context.Context, session *pb.Session) (*pb.Message, error) {
	log := klog.FromContext(ctx)
	if len(session.Messages) == 0 {
		return nil, fmt.Errorf("no messages in session")
	}
	m := b.client.GenerativeModel(b.model)

	// Add exec tool
	m.Tools = []*genai.Tool{
		{
			FunctionDeclarations: []*genai.FunctionDeclaration{
				{
					Name:        "exec",
					Description: "Execute a shell command in the sandbox",
					Parameters: &genai.Schema{
						Type: genai.TypeObject,
						Properties: map[string]*genai.Schema{
							"command": {
								Type:        genai.TypeString,
								Description: "The command to execute",
							},
						},
						Required: []string{"command"},
					},
				},
			},
		},
	}

	// Look for system message
	var systemInstruction []genai.Part
	for _, msg := range session.Messages {
		if msg.Role == "system" {
			for _, part := range msg.Parts {
				if text, ok := part.Data.(*pb.Part_Text); ok {
					systemInstruction = append(systemInstruction, genai.Text(text.Text))
				} else {
					log.Info("ignoring non-text part in system instruction", "type", fmt.Sprintf("%T", part.Data))
				}
			}
		}
	}
	if len(systemInstruction) > 0 {
		m.SystemInstruction = &genai.Content{
			Parts: systemInstruction,
		}
	}

	cs := m.StartChat()

	var history []*genai.Content
	for i := 0; i < len(session.Messages)-1; i++ {
		msg := session.Messages[i]
		if msg.Role == "system" {
			continue
		}
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}
		if role == "tool" {
			role = "function"
		}

		history = append(history, &genai.Content{
			Role:  role,
			Parts: toGenaiParts(ctx, msg.Parts),
		})
	}
	cs.History = history

	lastMsg := session.Messages[len(session.Messages)-1]
	resp, err := cs.SendMessage(ctx, toGenaiParts(ctx, lastMsg.Parts)...)
	if err != nil {
		return nil, err
	}

	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in response")
	}

	var pbParts []*pb.Part
	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			pbParts = append(pbParts, &pb.Part{
				Data: &pb.Part_Text{
					Text: string(text),
				},
			})
		} else if fc, ok := part.(genai.FunctionCall); ok {
			args, err := structpb.NewStruct(fc.Args)
			if err != nil {
				return nil, err
			}
			pbParts = append(pbParts, &pb.Part{
				Data: &pb.Part_FunctionCall{
					FunctionCall: &pb.FunctionCall{
						Name: fc.Name,
						Args: args,
					},
				},
			})
		} else {
			log.Info("unknown genai part type", "type", fmt.Sprintf("%T", part))
		}
	}

	return &pb.Message{
		Role:      "model",
		Parts:     pbParts,
		Timestamp: timestamppb.Now(),
	}, nil
}

func toGenaiParts(ctx context.Context, pbParts []*pb.Part) []genai.Part {
	log := klog.FromContext(ctx)
	var parts []genai.Part
	for _, p := range pbParts {
		switch part := p.Data.(type) {
		case *pb.Part_Text:
			parts = append(parts, genai.Text(part.Text))
		case *pb.Part_FunctionCall:
			parts = append(parts, genai.FunctionCall{
				Name: part.FunctionCall.Name,
				Args: part.FunctionCall.Args.AsMap(),
			})
		case *pb.Part_FunctionResponse:
			parts = append(parts, genai.FunctionResponse{
				Name:     part.FunctionResponse.Name,
				Response: part.FunctionResponse.Response.AsMap(),
			})
		default:
			log.Info("unknown bema part type", "type", fmt.Sprintf("%T", p.Data))
		}
	}
	return parts
}

func (b *GeminiBackend) Close() error {
	return b.client.Close()
}
