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
			systemInstruction = append(systemInstruction, genai.Text(msg.Content))
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

		parts := []genai.Part{}
		if msg.Content != "" {
			parts = append(parts, genai.Text(msg.Content))
		}

		if msg.ToolCalls != nil {
			if fcs, ok := msg.ToolCalls.Fields["functionCalls"]; ok {
				for _, fcValue := range fcs.GetListValue().Values {
					fc := fcValue.GetStructValue()
					name := fc.Fields["name"].GetStringValue()
					args := fc.Fields["args"].GetStructValue().AsMap()
					parts = append(parts, genai.FunctionCall{
						Name: name,
						Args: args,
					})
				}
			}
		}

		if msg.ToolOutputs != nil {
			if frs, ok := msg.ToolOutputs.Fields["functionResponses"]; ok {
				for _, frValue := range frs.GetListValue().Values {
					fr := frValue.GetStructValue()
					name := fr.Fields["name"].GetStringValue()
					// Genai expects response to be a map
					response := fr.AsMap()
					parts = append(parts, genai.FunctionResponse{
						Name:     name,
						Response: response,
					})
				}
			}
		}

		history = append(history, &genai.Content{
			Role:  role,
			Parts: parts,
		})
	}
	cs.History = history

	lastMsg := session.Messages[len(session.Messages)-1]
	resp, err := cs.SendMessage(ctx, genai.Text(lastMsg.Content))
	if err != nil {
		return nil, err
	}

	if len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates in response")
	}

	content := ""
	var toolCalls *structpb.Struct
	var functionCalls []any

	for _, part := range resp.Candidates[0].Content.Parts {
		if text, ok := part.(genai.Text); ok {
			content += string(text)
		}
		if fc, ok := part.(genai.FunctionCall); ok {
			functionCalls = append(functionCalls, map[string]any{
				"name": fc.Name,
				"args": fc.Args,
			})
		}
	}

	if len(functionCalls) > 0 {
		var err error
		toolCalls, err = structpb.NewStruct(map[string]any{
			"functionCalls": functionCalls,
		})
		if err != nil {
			return nil, err
		}
	}

	return &pb.Message{
		Role:      "assistant",
		Content:   content,
		ToolCalls: toolCalls,
		Timestamp: timestamppb.Now(),
	}, nil
}

func (b *GeminiBackend) Close() error {
	return b.client.Close()
}
