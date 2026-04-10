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
	"crypto/rand"
	"flag"
	"fmt"
	"log"
	"math"
	mrand "math/rand"
	"time"

	"github.com/bradfitz/gomemcache/memcache"
)

var (
	server      = flag.String("server", "localhost:11211", "Memcached server address")
	valueSize   = flag.Int("value-size", 512*1024, "Size of the values to write")
	period      = flag.Duration("period", 10*time.Minute, "Period of the sinusoidal load")
	minOps      = flag.Float64("min-ops", 10.0, "Minimum operations per second")
	maxOps      = flag.Float64("max-ops", 50.0, "Maximum operations per second")
	writeProb   = flag.Float64("write-prob", 0.2, "Probability of writing a new key")
	decayLambda = flag.Float64("decay-lambda", 10.0, "Lambda parameter for exponential decay of key age")
)

func main() {
	flag.Parse()

	mc := memcache.New(*server)
	val := make([]byte, *valueSize)

	var maxKeyIndex int64 = 0

	startTime := time.Now()

	for {
		now := time.Now()
		elapsed := now.Sub(startTime).Seconds()

		// Sinusoidal load level
		// load is between 0 and 1
		load := (math.Sin(2*math.Pi*elapsed/period.Seconds()) + 1.0) / 2.0
		currentOps := *minOps + load*(*maxOps-*minOps)
		sleepDuration := time.Duration(float64(time.Second) / currentOps)

		if mrand.Float64() < *writeProb {
			// Write a new key
			maxKeyIndex++
			key := fmt.Sprintf("key-%d", maxKeyIndex)
			rand.Read(val)
			err := mc.Set(&memcache.Item{Key: key, Value: val})
			if err != nil {
				log.Printf("Failed to write new key %s: %v", key, err)
			}
		} else {
			// Read an existing key
			if maxKeyIndex > 0 {
				// Exponential decay for age
				age := mrand.ExpFloat64() / *decayLambda

				// We map age to an index delta.  Lambda=10 means mean age is 0.1
				// Let's scale it so age=0.1 means ~100 keys back.
				delta := int64(age * 1000.0)
				if delta > maxKeyIndex-1 {
					delta = maxKeyIndex - 1
				}
				if delta < 0 {
					delta = 0
				}

				targetIndex := maxKeyIndex - delta
				key := fmt.Sprintf("key-%d", targetIndex)

				_, err := mc.Get(key)
				if err == memcache.ErrCacheMiss {
					// Read miss: write the value
					rand.Read(val)
					err := mc.Set(&memcache.Item{Key: key, Value: val})
					if err != nil {
						log.Printf("Failed to write after read miss %s: %v", key, err)
					}
				} else if err != nil {
					log.Printf("Failed to read %s: %v", key, err)
				}
			}
		}

		time.Sleep(sleepDuration)
	}
}
