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
	"crypto/sha256"
	"encoding/hex"
	"flag"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/klog/v2"

	pb "github.com/gke-labs/generation-ai/vxpu/pkg/api/v1alpha1"
)

type server struct {
	pb.UnimplementedExecutorServer
	clientset   kubernetes.Interface
	namespace   string
	imageName   string
	accelerator string

	mu          sync.RWMutex
	peerToModel map[string]string // maps peer IP:port -> modelKey
}

func getPeerKey(ctx context.Context) string {
	if p, ok := peer.FromContext(ctx); ok {
		return p.Addr.String()
	}
	return "unknown"
}

func (s *server) getOrStartPod(ctx context.Context, modelID string) (string, error) {
	log := klog.FromContext(ctx)
	if modelID == "" {
		return "", fmt.Errorf("modelID cannot be empty")
	}

	podName := "vxpu-executor-" + modelID

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
			// Give Kubernetes a moment to clean up
			time.Sleep(2 * time.Second)
		}
	} else if !errors.IsNotFound(err) {
		return "", fmt.Errorf("failed to get pod: %w", err)
	}

	// Create a new Pod if it didn't exist or was deleted
	_, err = s.clientset.CoreV1().Pods(s.namespace).Get(ctx, podName, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		log.Info("Creating a new executor pod", "pod", podName, "image", s.imageName, "accelerator", s.accelerator)

		// Build pod spec dynamically based on whether accelerator is configured
		var nodeSelector map[string]string
		var resources corev1.ResourceRequirements
		var containerEnv []corev1.EnvVar

		if s.accelerator != "" && s.accelerator != "none" {
			nodeSelector = map[string]string{
				"cloud.google.com/gke-accelerator": s.accelerator,
			}
			resources = corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:              resource.MustParse("4"),
					corev1.ResourceMemory:           resource.MustParse("20Gi"),
					corev1.ResourceEphemeralStorage: resource.MustParse("60Gi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceMemory:                 resource.MustParse("24Gi"),
					corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
				},
			}
			containerEnv = []corev1.EnvVar{
				{
					Name:  "LD_LIBRARY_PATH",
					Value: "/usr/local/nvidia/lib64",
				},
			}
		} else {
			// CPU/Kind friendly configuration
			resources = corev1.ResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceCPU:    resource.MustParse("500m"),
					corev1.ResourceMemory: resource.MustParse("1Gi"),
				},
				Limits: corev1.ResourceList{
					corev1.ResourceMemory: resource.MustParse("2Gi"),
				},
			}
		}

		newPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: s.namespace,
				Labels: map[string]string{
					"app":      "vxpu-executor",
					"model-id": modelID,
				},
			},
			Spec: corev1.PodSpec{
				NodeSelector: nodeSelector,
				Volumes: []corev1.Volume{
					{
						Name: "work",
						VolumeSource: corev1.VolumeSource{
							EmptyDir: &corev1.EmptyDirVolumeSource{
								SizeLimit: func() *resource.Quantity {
									q := resource.MustParse("80Gi")
									return &q
								}(),
							},
						},
					},
				},
				Containers: []corev1.Container{
					{
						Name:  "executor",
						Image: s.imageName,
						Ports: []corev1.ContainerPort{
							{
								ContainerPort: 50051,
							},
						},
						ReadinessProbe: &corev1.Probe{
							ProbeHandler: corev1.ProbeHandler{
								TCPSocket: &corev1.TCPSocketAction{
									Port: intstr.FromInt32(50051),
								},
							},
							InitialDelaySeconds: 3,
							PeriodSeconds:       2,
						},
						VolumeMounts: []corev1.VolumeMount{
							{
								Name:      "work",
								MountPath: "/tmp/vxpu",
							},
						},
						Env:       containerEnv,
						Resources: resources,
					},
				},
				RestartPolicy: corev1.RestartPolicyNever,
			},
		}

		_, err = s.clientset.CoreV1().Pods(s.namespace).Create(ctx, newPod, metav1.CreateOptions{})
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

func (s *server) LoadModel(ctx context.Context, req *pb.LoadModelRequest) (*pb.LoadModelResponse, error) {
	log := klog.FromContext(ctx)

	// Derive modelKey from manifest_json
	h := sha256.Sum256([]byte(req.ManifestJson))
	modelKey := hex.EncodeToString(h[:16])
	log.Info("LoadModel request received", "model_key", modelKey)

	peerKey := getPeerKey(ctx)
	if peerKey != "unknown" {
		s.mu.Lock()
		s.peerToModel[peerKey] = modelKey
		s.mu.Unlock()
	}

	podIP, err := s.getOrStartPod(ctx, modelKey)
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

	client := pb.NewExecutorClient(conn)

	var loadErr error
	for i := 0; i < 30; i++ {
		loadCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
		_, loadErr = client.LoadModel(loadCtx, req)
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

	log.Info("Successfully loaded model on executor", "model_key", modelKey)
	return &pb.LoadModelResponse{Success: true}, nil
}

func (s *server) NewSession(ctx context.Context, req *pb.NewSessionRequest) (*pb.NewSessionResponse, error) {
	log := klog.FromContext(ctx)
	peerKey := getPeerKey(ctx)

	s.mu.RLock()
	modelKey, exists := s.peerToModel[peerKey]
	s.mu.RUnlock()

	if !exists {
		return nil, status.Errorf(codes.FailedPrecondition, "no model has been loaded for this connection")
	}

	log.Info("NewSession request for", "model_key", modelKey)

	podIP, err := s.getOrStartPod(ctx, modelKey)
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

	client := pb.NewExecutorClient(conn)
	sessResp, err := client.NewSession(ctx, req)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create session on executor: %v", err)
	}

	wrappedSessionID := fmt.Sprintf("%s:%s", modelKey, sessResp.SessionId)
	log.Info("Successfully created routed session", "wrapped_session_id", wrappedSessionID)
	return &pb.NewSessionResponse{
		SessionId: wrappedSessionID,
	}, nil
}

func (s *server) Chat(ctx context.Context, req *pb.ChatRequest) (*pb.ChatResponse, error) {
	log := klog.FromContext(ctx)
	log.Info("Chat request received", "session_id", req.SessionId)

	parts := strings.SplitN(req.SessionId, ":", 2)
	if len(parts) != 2 {
		return nil, status.Errorf(codes.InvalidArgument, "invalid routed session ID format; expected 'modelKey:sessionID'")
	}
	modelKey, backendSessionID := parts[0], parts[1]

	podIP, err := s.getOrStartPod(ctx, modelKey)
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

	client := pb.NewExecutorClient(conn)
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
	imageFlag := flag.String("image", "", "The executor image to run")
	nsFlag := flag.String("namespace", "", "The Kubernetes namespace to use")
	acceleratorFlag := flag.String("accelerator", "", "GKE accelerator label for the executor pod (e.g., nvidia-l4)")
	flag.Parse()

	ctx := context.Background()
	log := klog.FromContext(ctx)

	// If image is not set via flag, fallback to environment variable
	imageName := *imageFlag
	if imageName == "" {
		imageName = os.Getenv("VXPU_EXECUTOR_IMAGE")
	}

	accelerator := *acceleratorFlag
	if accelerator == "" {
		accelerator = os.Getenv("VXPU_ACCELERATOR")
	}

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
			namespace = "default"
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

	pb.RegisterExecutorServer(s, &server{
		clientset:   clientset,
		namespace:   namespace,
		imageName:   imageName,
		accelerator: accelerator,
		peerToModel: make(map[string]string),
	})

	log.Info("Starting vxpu-router gRPC server", "port", *portFlag, "namespace", namespace, "image", imageName, "accelerator", accelerator)
	if err := s.Serve(lis); err != nil {
		log.Error(err, "failed to serve")
		os.Exit(1)
	}
}
