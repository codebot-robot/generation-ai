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

package blobstore

import "testing"

func TestOpenRejectsMisspeltLocations(t *testing.T) {
	for _, loc := range []string{"GS://bucket", "gs:/bucket", "gcs://bucket", "$(MODELSTORE_BLOB_STORE)", "relative/dir", ""} {
		if _, err := Open(t.Context(), loc); err == nil {
			t.Errorf("Open(%q) should fail", loc)
		}
	}
	if _, err := Open(t.Context(), t.TempDir()); err != nil {
		t.Errorf("absolute directory: %v", err)
	}
}
