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
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/experiments/model-manifest/pkg/api/v1alpha1"
)

type server struct {
	pb.UnimplementedModelRouterServer
	clientset kubernetes.Interface
	namespace string
	imageName string
}

func sanitizeModelID(modelID string) string {
	var res []rune
	for _, r := range strings.ToLower(modelID) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			res = append(res, r)
		} else {
			if len(res) > 0 && res[len(res)-1] != '-' {
				res = append(res, '-')
			}
		}
	}
	s := string(res)
	s = strings.Trim(s, "-")
	if len(s) > 63 {
		s = s[:63]
	}
	return s
}

// getOrStartPod retrieves the existing running pod IP or starts a new one and waits for it to become ready.
func (s *server) getOrStartPod(ctx context.Context, modelID string) (string, error) {
	log := klog.FromContext(ctx)
	if modelID == "" {
		return "", fmt.Errorf("modelID cannot be empty")
	}

	sanitized := sanitizeModelID(modelID)
	podName := "manifest-executor-" + sanitized

	// Check if pod already exists
	pod, err := s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
	if err == nil {
		log.Info("Found existing pod", "pod", podName, "phase", pod.Status.Phase)
		if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
			return pod.Status.PodIP, nil
		}
		if pod.Status.Phase == corev1.PodFailed || pod.Status.Phase == corev1.PodUnknown {
			log.Info("Existing pod is in bad state, deleting", "pod", podName, "phase", pod.Status.Phase)
			_ = s.clientset.CoreV1().Pods(s.namespace).Delete(ctx, podName, metav1.DeleteOptions{})
			time.Sleep(2 * time.Second)
		}
	} else if !errors.IsNotFound(err) {
		return "", fmt.Errorf("failed to get pod: %w", err)
	}

	// Create a new Pod if it didn't exist or was deleted
	_, err = s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		log.Info("Creating a new GPU pod", "pod", podName, "image", s.imageName)
		newPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: s.namespace,
				Labels: map[string]string{
					"app":      "manifest-executor",
					"variant":  "gpu",
					"model-id": sanitized,
				},
			},
			Spec: corev1.PodSpec{
				NodeSelector: map[string]string{
					"cloud.google.com/gke-accelerator": "nvidia-l4",
				},
				Containers: []corev1.Container{
					{
						Name:       "executor",
						Image:      s.imageName,
						Command:    []string{"python3", "src/serve_grpc.py", "--port", "50051"},
						WorkingDir: "/app",
						Env: []corev1.EnvVar{
							{
								Name:  "LD_LIBRARY_PATH",
								Value: "/usr/local/nvidia/lib64",
							},
						},
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("4"),
								corev1.ResourceMemory: resource.MustParse("16Gi"),
							},
							Limits: corev1.ResourceList{
								corev1.ResourceMemory:                 resource.MustParse("16Gi"),
								corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
							},
						},
					},
				},
				RestartPolicy: corev1.RestartPolicyNever,
			},
		}

		pod, err = s.clientset.CoreV1().Pods(s.namespace).Create(ctx, newPod, metav1.CreateOptions{})
		if err != nil {
			return "", fmt.Errorf("failed to create pod: %w", err)
		}
	}

	// Wait for the Pod to be running and have an IP
	log.Info("Waiting for pod to be running and get an IP", "pod", podName)
	var podIP string
	for i := 0; i < 300; i++ {
		pod, err = s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			log.Error(err, "failed to get pod status, retrying", "pod", podName)
		} else {
			if pod.Status.Phase == corev1.PodRunning && pod.Status.PodIP != "" {
				podIP = pod.Status.PodIP
				break
			}
			if pod.Status.Phase == corev1.PodFailed {
				return "", fmt.Errorf("pod failed to start")
			}
		}
		time.Sleep(2 * time.Second)
	}

	if podIP == "" {
		return "", fmt.Errorf("timed out waiting for pod to get an IP")
	}

	return podIP, nil
}

func (s *server) PrepareModel(ctx context.Context, req *pb.PrepareModelRequest) (*pb.PrepareModelResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("PrepareModel request", "model_id", req.ModelId)

	podIP, err := s.getOrStartPod(ctx, req.ModelId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}

	endpoint := fmt.Sprintf("%s:50051", podIP)
	log.Info("Executor pod ready. Loading model...", "endpoint", endpoint)

	maxMsgSize := 128 * 1024 * 1024
	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}
	defer conn.Close()

	client := pb.NewModelExecutorClient(conn)

	var loadErr error
	for i := 0; i < 30; i++ {
		loadCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		_, loadErr = client.LoadModel(loadCtx, &pb.LoadModelRequest{
			ManifestJson: req.ManifestJson,
			BindingJson:  req.BindingJson,
			PrefillGraph: req.PrefillGraph,
			DecodeGraph:  req.DecodeGraph,
		})
		cancel()
		if loadErr == nil {
			break
		}
		log.Info("Waiting for executor gRPC server to accept LoadModel", "attempt", i, "error", loadErr.Error())
		time.Sleep(2 * time.Second)
	}

	if loadErr != nil {
		return nil, status.Errorf(codes.Internal, "failed to load model on executor: %v", loadErr)
	}

	log.Info("Successfully prepared model", "model_id", req.ModelId)
	return &pb.PrepareModelResponse{
		PreparedModelId: req.ModelId,
	}, nil
}

func (s *server) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("NewSession request", "prepared_model_id", req.ModelId)

	podIP, err := s.getOrStartPod(ctx, req.ModelId)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}

	endpoint := fmt.Sprintf("%s:50051", podIP)
	maxMsgSize := 128 * 1024 * 1024
	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}
	defer conn.Close()

	client := pb.NewModelExecutorClient(conn)
	sessResp, err := client.NewSession(ctx, req)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create session on executor: %v", err)
	}

	// Encode routing info into the session ID: "modelID:sessionID"
	routedSessionID := fmt.Sprintf("%s:%s", req.ModelId, sessResp.SessionId)
	log.Info("Successfully created routed session", "routed_session_id", routedSessionID)
	return &pb.NewSessionResponse{
		SessionId: routedSessionID,
	}, nil
}

func (s *server) Chat(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("Chat request", "routed_session_id", req.SessionId)

	parts := strings.SplitN(req.SessionId, ":", 2)
	if len(parts) != 2 {
		return nil, status.Errorf(codes.InvalidArgument, "invalid routed session ID format; expected 'modelID:sessionID'")
	}
	modelID, backendSessionID := parts[0], parts[1]

	podIP, err := s.getOrStartPod(ctx, modelID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get or start pod: %v", err)
	}

	endpoint := fmt.Sprintf("%s:50051", podIP)
	maxMsgSize := 128 * 1024 * 1024
	conn, err := grpc.NewClient(
		endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(maxMsgSize),
			grpc.MaxCallSendMsgSize(maxMsgSize),
		),
	)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to connect to executor: %v", err)
	}
	defer conn.Close()

	client := pb.NewModelExecutorClient(conn)
	resp, err := client.Chat(ctx, &pb.ChatRequest{
		SessionId:    backendSessionID,
		Text:         req.Text,
		MaxNewTokens: req.MaxNewTokens,
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to execute chat on executor: %v", err)
	}

	return resp, nil
}

func main() {
	klog.InitFlags(nil)
	defer klog.Flush()

	portFlag := flag.Int("port", 50051, "The server port")
	imageFlag := flag.String("image", "gcr.io/justinsb-knotai-dev/generation-ai/manifest-executor:dev", "The executor image to run")
	nsFlag := flag.String("namespace", "", "The Kubernetes namespace to use")
	flag.Parse()

	ctx := context.Background()
	log := klog.FromContext(ctx)

	// Load Kubeconfig
	var config *rest.Config
	var err error
	if kubeconfig := os.Getenv("KUBECONFIG"); kubeconfig != "" {
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	} else if home := homedir.HomeDir(); home != "" {
		config, err = clientcmd.BuildConfigFromFlags("", filepath.Join(home, ".kube", "config"))
	} else {
		config, err = rest.InClusterConfig()
	}
	if err != nil {
		log.Error(err, "failed to load kubeconfig")
		os.Exit(1)
	}

	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		log.Error(err, "failed to create kubernetes clientset")
		os.Exit(1)
	}

	namespace := *nsFlag
	if namespace == "" {
		if nsBytes, err := os.ReadFile("/var/run/secrets/kubernetes.io/serviceaccount/namespace"); err == nil {
			namespace = string(nsBytes)
		} else {
			log.Error(nil, "namespace is required if it cannot be read from serviceaccount secrets. Set --namespace flag or KUBECONFIG.")
			os.Exit(1)
		}
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *portFlag))
	if err != nil {
		log.Error(err, "failed to listen")
		os.Exit(1)
	}

	maxMsgSize := 128 * 1024 * 1024
	s := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxMsgSize),
		grpc.MaxSendMsgSize(maxMsgSize),
	)

	pb.RegisterModelRouterServer(s, &server{
		clientset: clientset,
		namespace: namespace,
		imageName: *imageFlag,
	})

	log.Info("Starting model-router gRPC server", "port", *portFlag, "namespace", namespace, "image", *imageFlag)
	if err := s.Serve(lis); err != nil {
		log.Error(err, "failed to serve")
		os.Exit(1)
	}
}
