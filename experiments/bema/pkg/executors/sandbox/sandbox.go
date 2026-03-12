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
	"bytes"
	"context"
	"fmt"
	"os"
	"time"

	pb "github.com/gke-labs/generation-ai/experiments/bema/pkg/api/v1alpha1"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
)

var (
	agentSandboxGVR = schema.GroupVersionResource{
		Group:    "agents.x-k8s.io",
		Version:  "v1alpha1",
		Resource: "sandboxes",
	}
)

type SandboxExecutor struct {
	client     dynamic.Interface
	kubeClient kubernetes.Interface
	config     *rest.Config
	namespace  string
}

func New(ctx context.Context) (*SandboxExecutor, error) {
	restConfig, err := config.GetConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get kubernetes config: %v", err)
	}

	client, err := dynamic.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %v", err)
	}

	kubeClient, err := kubernetes.NewForConfig(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %v", err)
	}

	ns := os.Getenv("NAMESPACE")
	if ns == "" {
		ns = "bema-sandboxes"
	}

	return &SandboxExecutor{
		client:     client,
		kubeClient: kubeClient,
		config:     restConfig,
		namespace:  ns,
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

	_, err := e.client.Resource(agentSandboxGVR).Namespace(e.namespace).Get(ctx, name, metav1.GetOptions{})
	if err == nil {
		return e.waitForSandboxReady(ctx, name)
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

	_, err = e.client.Resource(agentSandboxGVR).Namespace(e.namespace).Create(ctx, sandbox, metav1.CreateOptions{})
	if err != nil {
		return fmt.Errorf("failed to create Sandbox: %v", err)
	}

	return e.waitForSandboxReady(ctx, name)
}

func (e *SandboxExecutor) waitForSandboxReady(ctx context.Context, name string) error {
	return wait.PollUntilContextTimeout(ctx, 1*time.Second, 60*time.Second, true, func(ctx context.Context) (bool, error) {
		sandbox, err := e.client.Resource(agentSandboxGVR).Namespace(e.namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, nil // Continue polling
		}

		conditions, found, err := unstructured.NestedSlice(sandbox.Object, "status", "conditions")
		if err != nil || !found {
			return false, nil
		}

		for _, c := range conditions {
			condition, ok := c.(map[string]any)
			if !ok {
				continue
			}
			if condition["type"] == "Ready" && condition["status"] == "True" {
				return true, nil
			}
		}

		return false, nil
	})
}

func (e *SandboxExecutor) executeInSandbox(ctx context.Context, sessionID string, command string) (string, error) {
	if err := e.ensureSandbox(ctx, sessionID); err != nil {
		return "", err
	}

	name := "bema-" + sessionID

	req := e.kubeClient.CoreV1().RESTClient().Post().
		Resource("pods").
		Name(name).
		Namespace(e.namespace).
		SubResource("exec")

	option := &corev1.PodExecOptions{
		Container: "sandbox",
		Command:   []string{"sh", "-c", command},
		Stdin:     false,
		Stdout:    true,
		Stderr:    true,
		TTY:       false,
	}

	req.VersionedParams(
		option,
		scheme.ParameterCodec,
	)

	exec, err := remotecommand.NewSPDYExecutor(e.config, "POST", req.URL())
	if err != nil {
		return "", fmt.Errorf("failed to create SPDY executor: %v", err)
	}

	var stdout, stderr bytes.Buffer
	err = exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &stdout,
		Stderr: &stderr,
	})

	output := stdout.String()
	if stderr.Len() > 0 {
		if output != "" {
			output += "\n"
		}
		output += stderr.String()
	}

	return output, err
}
