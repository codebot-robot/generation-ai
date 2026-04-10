# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import datetime

# Search for jsonl files
artifacts_dir = os.environ.get("ARTIFACTS", "artifacts")
if not os.path.exists(artifacts_dir):
    artifacts_dir = "." # fallback

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ScalingPolicy Metrics Visualization</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: sans-serif; margin: 20px; background-color: #fafafa; }
        .dashboard { display: flex; flex-wrap: wrap; gap: 20px; }
        .scenario-container { border: 1px solid #ddd; padding: 15px; border-radius: 8px; background-color: white; margin-bottom: 20px; }
        .plot-container { flex: 1 1 500px; height: 400px; min-width: 0; max-width: 100%; }
        .header { text-align: center; padding-bottom: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>ScalingPolicy Performance Metrics</h1>
        <p>Comparing Memory Management Strategies</p>
    </div>
    <div id="dashboard"></div>
    <script>
        async function loadData() {
            const response = await fetch('/api/data');
            const data = await response.json();
            
            const dashboard = document.getElementById('dashboard');
            
            for (const scenario in data) {
                const sData = data[scenario];
                
                const sDiv = document.createElement('div');
                sDiv.className = 'scenario-container';
                const sTitle = document.createElement('h2');
                sTitle.innerText = `Scenario: ${scenario}`;
                sDiv.appendChild(sTitle);
                
                const plotsDiv = document.createElement('div');
                plotsDiv.className = 'dashboard';
                sDiv.appendChild(plotsDiv);
                dashboard.appendChild(sDiv);

                // 1. Memory Plot
                const memTraces = [];
                for (const metric of ['pod_memory_limit', 'pod_memory_request', 'memcached_bytes']) {
                    if (!sData[metric]) continue;
                    for (const pod in sData[metric]) {
                        if (pod.includes("client")) continue; // Only server metrics
                        
                        let color = null;
                        let line_dash = null;
                        let name = metric;
                        
                        if (metric === 'pod_memory_limit') { color = 'red'; line_dash = 'dash'; name = 'Pod Limit'; }
                        if (metric === 'pod_memory_request') { color = 'orange'; line_dash = 'dot'; name = 'Pod Request'; }
                        if (metric === 'memcached_bytes') { color = 'purple'; name = 'App Bytes Used'; }
                        
                        if (pod) {
                            name = `${name} (${pod})`;
                        }
                        
                        memTraces.push({
                            x: sData[metric][pod].times,
                            y: sData[metric][pod].values,
                            mode: 'lines',
                            name: name,
                            line: { color: color, dash: line_dash }
                        });
                    }
                }
                
                if (memTraces.length > 0) {
                    const div = document.createElement('div');
                    div.className = 'plot-container';
                    plotsDiv.appendChild(div);
                    Plotly.newPlot(div, memTraces, {
                        title: 'Memory Profile',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Memory (Bytes)', rangemode: 'tozero' }
                    }, {responsive: true});
                }
                
                // 2. Cache Metrics Plot
                const cacheTraces = [];
                const hits = sData['memcached_get_hits'];
                const misses = sData['memcached_get_misses'];
                
                if (hits && misses) {
                    // Find the keys (pods)
                    const podKey = Object.keys(hits)[0] || '';
                    let hitTimes = hits[podKey].times;
                    let hitVals = hits[podKey].values;
                    let missVals = misses[podKey].values;
                    
                    let hitRatioVals = [];
                    for(let i=0; i<hitVals.length; i++) {
                        let total = hitVals[i] + missVals[i];
                        hitRatioVals.push(total > 0 ? (hitVals[i] / total) * 100 : 0);
                    }
                    
                    cacheTraces.push({
                        x: hitTimes,
                        y: hitRatioVals,
                        mode: 'lines',
                        name: `Hit Ratio %`,
                        line: { color: 'green' }
                    });
                    
                    const div2 = document.createElement('div');
                    div2.className = 'plot-container';
                    plotsDiv.appendChild(div2);
                    Plotly.newPlot(div2, cacheTraces, {
                        title: 'Cache Hit Ratio',
                        xaxis: { title: 'Time' },
                        yaxis: { title: 'Hit Ratio (%)', range: [0, 100] }
                    }, {responsive: true});
                }
            }
        }
        loadData();
    </script>
</body>
</html>
"""

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))
        elif self.path == '/api/data':
            data = self.load_metrics()
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            
    def load_metrics(self):
        data = {}
        
        for root, dirs, files in os.walk(artifacts_dir):
            for file in files:
                if file.endswith('.jsonl'):
                    filepath = os.path.join(root, file)
                    parts = os.path.relpath(root, artifacts_dir).split(os.sep)
                    scenario = parts[-1] if len(parts) > 0 else 'Unknown'
                    if scenario == 'Unknown' or scenario == '.':
                        scenario = 'Default'
                    
                    if scenario not in data:
                        data[scenario] = {}
                        
                    with open(filepath, 'r') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                                scope_metrics = record.get('scopeMetrics', [])
                                for sm in scope_metrics:
                                    metrics = sm.get('metrics', [])
                                    for m in metrics:
                                        name = m.get('name')
                                        if name not in data[scenario]:
                                            data[scenario][name] = {}
                                            
                                        gauge = m.get('gauge', {})
                                        dps = gauge.get('dataPoints', [])
                                        for dp in dps:
                                            ts_nano = int(dp.get('timeUnixNano', 0))
                                            ts_sec = ts_nano / 1e9
                                            ts_iso = datetime.datetime.utcfromtimestamp(ts_sec).isoformat() + 'Z'
                                            
                                            val = dp.get('asDouble', 0.0)
                                            
                                            pod = ''
                                            for attr in dp.get('attributes', []):
                                                if attr.get('key') == 'pod':
                                                    pod = attr.get('value', {}).get('stringValue', '')
                                                    
                                            if pod not in data[scenario][name]:
                                                data[scenario][name][pod] = {'times': [], 'values': []}
                                                
                                            data[scenario][name][pod]['times'].append(ts_iso)
                                            data[scenario][name][pod]['values'].append(val)
                            except Exception as e:
                                pass
        
        for scenario in data:
            for metric in data[scenario]:
                for pod in data[scenario][metric]:
                    times = data[scenario][metric][pod]['times']
                    vals = data[scenario][metric][pod]['values']
                    if times:
                        sorted_pairs = sorted(zip(times, vals))
                        data[scenario][metric][pod]['times'] = [p[0] for p in sorted_pairs]
                        data[scenario][metric][pod]['values'] = [p[1] for p in sorted_pairs]
                        
        return data

def run(server_class=HTTPServer, handler_class=MetricsHandler, port=8000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    print(f"Scanning directory: {os.path.abspath(artifacts_dir)}")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
