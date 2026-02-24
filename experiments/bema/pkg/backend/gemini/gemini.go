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
	"google.golang.org/genai"
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
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey: apiKey,
	})
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

	// Add exec tool
	tools := []*genai.Tool{
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

	config := &genai.GenerateContentConfig{
		Tools: tools,
	}

	// Look for system message
	var systemInstructionParts []*genai.Part
	for _, msg := range session.Messages {
		if msg.Role == "system" {
			for _, part := range msg.Parts {
				if text := part.GetText(); text != "" {
					systemInstructionParts = append(systemInstructionParts, &genai.Part{Text: text})
				} else {
					log.Info("ignoring non-text part in system instruction", "type", fmt.Sprintf("%T", part.Data))
				}
			}
		}
	}
	if len(systemInstructionParts) > 0 {
		config.SystemInstruction = &genai.Content{
			Parts: systemInstructionParts,
		}
	}

	var contents []*genai.Content
	for _, msg := range session.Messages {
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

		contents = append(contents, &genai.Content{
			Role:  role,
			Parts: toGenaiParts(ctx, msg.Parts),
		})
	}

	resp, err := b.client.Models.GenerateContent(ctx, b.model, contents, config)
	if err != nil {
		return nil, err
	}

	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in response")
	}

	var pbParts []*pb.Part
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.Text != "" {
			pbParts = append(pbParts, &pb.Part{
				Data: &pb.Part_Text{
					Text: part.Text,
				},
				Thought:          part.Thought,
				ThoughtSignature: part.ThoughtSignature,
			})
		} else if fc := part.FunctionCall; fc != nil {
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
				Thought:          part.Thought,
				ThoughtSignature: part.ThoughtSignature,
			})
		} else if part.Thought {
			pbParts = append(pbParts, &pb.Part{
				Thought:          true,
				ThoughtSignature: part.ThoughtSignature,
			})
		} else {
			log.Info("unknown genai part", "part", part)
		}
	}

	return &pb.Message{
		Role:      "model",
		Parts:     pbParts,
		Timestamp: timestamppb.Now(),
	}, nil
}

func toGenaiParts(ctx context.Context, pbParts []*pb.Part) []*genai.Part {
	log := klog.FromContext(ctx)
	var parts []*genai.Part
	for _, p := range pbParts {
		part := &genai.Part{
			Thought:          p.Thought,
			ThoughtSignature: p.ThoughtSignature,
		}
		switch data := p.Data.(type) {
		case *pb.Part_Text:
			part.Text = data.Text
		case *pb.Part_FunctionCall:
			part.FunctionCall = &genai.FunctionCall{
				Name: data.FunctionCall.Name,
				Args: data.FunctionCall.Args.AsMap(),
			}
		case *pb.Part_FunctionResponse:
			part.FunctionResponse = &genai.FunctionResponse{
				Name:     data.FunctionResponse.Name,
				Response: data.FunctionResponse.Response.AsMap(),
			}
		default:
			if !p.Thought {
				log.Info("unknown bema part type", "type", fmt.Sprintf("%T", p.Data))
			}
		}
		parts = append(parts, part)
	}
	return parts
}

func (b *GeminiBackend) Close() error {
	return nil
}
