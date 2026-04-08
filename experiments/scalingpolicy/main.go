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
	"os"

	"github.com/google/cel-go/cel"
	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	scalingpolicyv1alpha1 "github.com/gke-labs/generation-ai/experiments/scalingpolicy/pkg/api/v1alpha1"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(scalingpolicyv1alpha1.AddToScheme(scheme))
}

type ScalingPolicyReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

func (r *ScalingPolicyReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := ctrl.Log.WithName("scalingpolicy-controller")

	var scalingPolicy scalingpolicyv1alpha1.ScalingPolicy
	if err := r.Get(ctx, req.NamespacedName, &scalingPolicy); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	log.Info("Reconciling ScalingPolicy", "name", scalingPolicy.Name)

	if scalingPolicy.Spec.Target.Kind == "Deployment" {
		var deploy appsv1.Deployment
		if err := r.Get(ctx, types.NamespacedName{
			Name:      scalingPolicy.Spec.Target.Name,
			Namespace: req.Namespace,
		}, &deploy); err != nil {
			return ctrl.Result{}, client.IgnoreNotFound(err)
		}

		for _, val := range scalingPolicy.Spec.Values {
			if val.Path == "spec.replicas" {
				env, err := cel.NewEnv()
				if err != nil {
					log.Error(err, "failed to create CEL env")
					continue
				}
				ast, iss := env.Compile(val.Expression)
				if iss.Err() != nil {
					log.Error(iss.Err(), "failed to compile CEL expression")
					continue
				}
				prg, err := env.Program(ast)
				if err != nil {
					log.Error(err, "failed to create CEL program")
					continue
				}
				out, _, err := prg.Eval(map[string]any{})
				if err != nil {
					log.Error(err, "failed to evaluate CEL expression")
					continue
				}

				var replicasInt32 int32
				switch v := out.Value().(type) {
				case int64:
					replicasInt32 = int32(v)
				case int:
					replicasInt32 = int32(v)
				default:
					log.Error(nil, "CEL expression did not evaluate to integer", "type", out.Type())
					continue
				}

				if val.Min != nil && replicasInt32 < *val.Min {
					replicasInt32 = *val.Min
				}
				if val.Max != nil && replicasInt32 > *val.Max {
					replicasInt32 = *val.Max
				}

				if deploy.Spec.Replicas == nil || *deploy.Spec.Replicas != replicasInt32 {
					deploy.Spec.Replicas = &replicasInt32
					if err := r.Update(ctx, &deploy); err != nil {
						log.Error(err, "failed to update deployment replicas")
						return ctrl.Result{}, err
					}
					log.Info("Updated Deployment replicas", "replicas", replicasInt32)
				}
			}
		}
	}

	return ctrl.Result{}, nil
}
func main() {
	var metricsAddr string
	flag.StringVar(&metricsAddr, "metrics-addr", ":8080", "The address the metric endpoint binds to.")
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseDevMode(true)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
		Scheme: scheme,
	})
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	if err = (&ScalingPolicyReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "ScalingPolicy")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

func (r *ScalingPolicyReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&scalingpolicyv1alpha1.ScalingPolicy{}).
		Complete(r)
}
