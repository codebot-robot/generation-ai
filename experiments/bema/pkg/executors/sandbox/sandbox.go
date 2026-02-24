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

package sandbox

import (
	"context"
	"fmt"
	"os"
	"os/exec"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
)

var (
	agentSandboxGVR = schema.GroupVersionResource{
		Group:    "agents.x-k8s.io",
		Version:  "v1alpha1",
		Resource: "sandboxes",
	}
)

type SandboxExecutor struct {
	client    dynamic.Interface
	namespace string
}

func New(ctx context.Context) (*SandboxExecutor, error) {
	restConfig, err := rest.InClusterConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get kubernetes config: %v", err)
	}

	client, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %v", err)
	}

	ns := os.Getenv("NAMESPACE")
	if ns == "" {
		ns = "bema-sandboxes"
	}

	return &SandboxExecutor{
		client:    client,
		namespace: ns,
	}, nil
}

func (e *SandboxExecutor) Execute(ctx context.Context, sessionID string, message *pb.Message) (*pb.Message, error) {
	var pbParts []*pb.Part

	for _, p := range message.Parts {
		fcPart, ok := p.Data.(*pb.Part_FunctionCall)
		if !ok {
			continue
		}
		fc := fcPart.FunctionCall
		name := fc.Name
		args := fc.Args

		if name == "exec" {
			command := args.Fields["command"].GetStringValue()
			output, err := e.executeInSandbox(ctx, sessionID, command)

			result := map[string]any{
				"output": output,
			}
			if err != nil {
				result["error"] = err.Error()
			}
			responseStruct, err := structpb.NewStruct(result)
			if err != nil {
				return nil, err
			}

			pbParts = append(pbParts, &pb.Part{
				Data: &pb.Part_FunctionResponse{
					FunctionResponse: &pb.FunctionResponse{
						Name:     name,
						Response: responseStruct,
					},
				},
			})
		} else {
			responseStruct, err := structpb.NewStruct(map[string]any{
				"error": "unknown tool",
			})
			if err != nil {
				return nil, err
			}
			pbParts = append(pbParts, &pb.Part{
				Data: &pb.Part_FunctionResponse{
					FunctionResponse: &pb.FunctionResponse{
						Name:     name,
						Response: responseStruct,
					},
				},
			})
		}
	}

	if len(pbParts) == 0 {
		return nil, nil
	}

	return &pb.Message{
		Role:      "function",
		Parts:     pbParts,
		Timestamp: timestamppb.Now(),
	}, nil
}

func (e *SandboxExecutor) ensureSandbox(ctx context.Context, sessionID string) error {
	name := "bema-" + sessionID

	_, err := e.client.Resource(agentSandboxGVR).Namespace(e.namespace).Get(ctx, name, v1.GetOptions{})
	if err == nil {
		return nil
	}

	// Create it
	sandbox := &unstructured.Unstructured{
		Object: map[string]any{
			"apiVersion": "agents.x-k8s.io/v1alpha1",
			"kind":       "Sandbox",
			"metadata": map[string]any{
				"name": name,
			},
			"spec": map[string]any{
				"podTemplate": map[string]any{
					"spec": map[string]any{
						"containers": []any{
							map[string]any{
								"name":    "sandbox",
								"image":   "debian:latest",
								"command": []string{"sleep", "infinity"},
							},
						},
					},
				},
			},
		},
	}

	_, err = e.client.Resource(agentSandboxGVR).Namespace(e.namespace).Create(ctx, sandbox, v1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create Sandbox: %v", err)
	}

	// Wait for it to be ready
	cmd := exec.CommandContext(ctx, "kubectl", "wait", "--for=condition=Ready", "sandbox", "-n", e.namespace, name, "--timeout=60s")
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to wait for Sandbox readiness: %v: %s", err, string(output))
	}

	return nil
}

func (e *SandboxExecutor) executeInSandbox(ctx context.Context, sessionID string, command string) (string, error) {
	if err := e.ensureSandbox(ctx, sessionID); err != nil {
		return "", err
	}

	name := "bema-" + sessionID

	// We use kubectl exec.
	// Since we are using Sandbox, we should try to exec into the pod created by it.
	// We'll assume for now that the pod has the same name or we can find it by label.

	cmd := exec.CommandContext(ctx, "kubectl", "exec", "-n", e.namespace, name, "--", "sh", "-c", command)
	output, err := cmd.CombinedOutput()

	return string(output), err
}
