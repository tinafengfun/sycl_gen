#!/usr/bin/env python3
"""
Benchmark Monitoring System
Provides web dashboard and progress tracking for kernel testing
"""

import json
import time
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import socket

class TestState:
    """Global test state manager"""
    def __init__(self):
        self.state_file = Path("test_state.json")
        self.results_file = Path("test_results.json")
        self.log_file = Path("test_log.txt")
        
        # Test configuration
        self.kernels = ["softmax", "global_avg_pool", "winograd_input_transform"]
        self.versions = ["V0", "V1", "V2", "V3", "V4", "V5"]
        self.sizes = [256, 512, 1024, 4096, 16384]
        
        # Calculate total tests
        self.total_tests = len(self.kernels) * len(self.versions) * len(self.sizes)
        self.completed_tests = 0
        self.failed_tests = 0
        
        # Current status
        self.current_kernel = ""
        self.current_version = ""
        self.current_size = 0
        self.current_iteration = 0
        self.status = "INITIALIZING"  # INITIALIZING, RUNNING, PAUSED, COMPLETED, ERROR
        
        # Timing
        self.start_time = None
        self.current_test_start = None
        
        # Results storage
        self.results = []
        self.errors = []
        
        # Load existing state if present
        self.load_state()
    
    def load_state(self):
        """Load previous state if exists"""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                self.completed_tests = data.get("completed_tests", 0)
                self.failed_tests = data.get("failed_tests", 0)
                self.results = data.get("results", [])
                self.errors = data.get("errors", [])
    
    def save_state(self):
        """Save current state to file"""
        data = {
            "completed_tests": self.completed_tests,
            "failed_tests": self.failed_tests,
            "current_kernel": self.current_kernel,
            "current_version": self.current_version,
            "current_size": self.current_size,
            "status": self.status,
            "results": self.results,
            "errors": self.errors[-10:],  # Keep last 10 errors
            "timestamp": datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def log(self, message):
        """Log message to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")
    
    def start_test(self, kernel, version, size):
        """Mark start of a test"""
        self.current_kernel = kernel
        self.current_version = version
        self.current_size = size
        self.current_test_start = time.time()
        self.status = "RUNNING"
        self.log(f"Starting: {kernel}/{version}/size={size}")
        self.save_state()
    
    def complete_test(self, result):
        """Mark completion of a test"""
        self.completed_tests += 1
        self.results.append({
            "kernel": self.current_kernel,
            "version": self.current_version,
            "size": self.current_size,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        elapsed = time.time() - self.current_test_start
        self.log(f"Completed: {self.current_kernel}/{self.current_version}/size={self.current_size} in {elapsed:.1f}s")
        self.save_state()
    
    def fail_test(self, error):
        """Mark failure of a test"""
        self.failed_tests += 1
        self.errors.append({
            "kernel": self.current_kernel,
            "version": self.current_version,
            "size": self.current_size,
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
        self.log(f"FAILED: {self.current_kernel}/{self.current_version}/size={self.current_size} - {error}")
        self.save_state()
        self.status = "ERROR"
    
    def get_progress(self):
        """Get current progress info"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            if self.completed_tests > 0:
                avg_time_per_test = elapsed / self.completed_tests
                remaining_tests = self.total_tests - self.completed_tests
                eta = avg_time_per_test * remaining_tests
            else:
                eta = 0
        else:
            elapsed = 0
            eta = 0
        
        return {
            "total_tests": self.total_tests,
            "completed_tests": self.completed_tests,
            "failed_tests": self.failed_tests,
            "percent_complete": (self.completed_tests / self.total_tests) * 100,
            "current_kernel": self.current_kernel,
            "current_version": self.current_version,
            "current_size": self.current_size,
            "status": self.status,
            "elapsed_time": str(timedelta(seconds=int(elapsed))),
            "estimated_remaining": str(timedelta(seconds=int(eta))),
            "tests_per_hour": self.completed_tests / (elapsed / 3600) if elapsed > 0 else 0
        }

# Global state instance
state = TestState()

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for dashboard"""
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_html()
        elif self.path == '/api/status':
            self.send_json(state.get_progress())
        elif self.path == '/api/results':
            self.send_json(state.results)
        elif self.path == '/api/errors':
            self.send_json(state.errors)
        else:
            self.send_error(404)
    
    def send_html(self):
        """Send dashboard HTML"""
        progress = state.get_progress()
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>GPU Kernel Benchmark Dashboard</title>
    <meta http-equiv="refresh" content="30">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
        .header {{ background: #333; color: white; padding: 20px; border-radius: 5px; }}
        .status-box {{ background: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .progress-bar {{ width: 100%; height: 30px; background: #ddd; border-radius: 15px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: #4CAF50; transition: width 0.5s; }}
        .metric {{ display: inline-block; margin: 10px 20px; padding: 10px; background: #e3f2fd; border-radius: 5px; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1976d2; }}
        .current {{ background: #fff3e0; padding: 15px; border-radius: 5px; margin: 10px 0; }}
        .error {{ background: #ffebee; color: #c62828; padding: 10px; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
        tr:hover {{ background: #f5f5f5; }}
        .status-running {{ color: #4CAF50; font-weight: bold; }}
        .status-error {{ color: #f44336; font-weight: bold; }}
        .status-paused {{ color: #ff9800; font-weight: bold; }}
        .kernel-progress {{ margin: 10px 0; padding: 10px; background: white; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 GPU Kernel Benchmark Dashboard</h1>
        <p>Intel BMG B60 Optimization Project</p>
    </div>
    
    <div class="status-box">
        <h2>Overall Progress</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress['percent_complete']}%"></div>
        </div>
        <p style="text-align: center; margin-top: 10px;">
            {progress['completed_tests']} / {progress['total_tests']} tests ({progress['percent_complete']:.1f}%)
        </p>
        
        <div style="margin-top: 20px;">
            <div class="metric">
                <div class="metric-label">Status</div>
                <div class="metric-value status-{progress['status'].lower()}">{progress['status']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Elapsed Time</div>
                <div class="metric-value">{progress['elapsed_time']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">ETA</div>
                <div class="metric-value">{progress['estimated_remaining']}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Tests/Hour</div>
                <div class="metric-value">{progress['tests_per_hour']:.1f}</div>
            </div>
        </div>
    </div>
    
    <div class="current">
        <h3>🔄 Current Test</h3>
        <p><strong>Kernel:</strong> {progress['current_kernel'] or 'Waiting to start...'}</p>
        <p><strong>Version:</strong> {progress['current_version'] or '-'}</p>
        <p><strong>Size:</strong> {progress['current_size'] or '-'}</p>
    </div>
    
    <div class="status-box">
        <h3>📊 Results Summary</h3>
        <p>✅ Completed: {progress['completed_tests']}</p>
        <p>❌ Failed: {progress['failed_tests']}</p>
        <p>⏳ Remaining: {progress['total_tests'] - progress['completed_tests']}</p>
    </div>
    
    <div class="status-box">
        <h3>📝 Recent Log Entries</h3>
        <pre id="log-content">Loading...</pre>
    </div>
    
    <script>
        // Auto-refresh every 30 seconds
        setInterval(() => location.reload(), 30000);
        
        // Fetch log content
        fetch('/api/status')
            .then(r => r.json())
            .then(data => console.log('Status updated:', data));
    </script>
</body>
</html>'''
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def send_json(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def start_dashboard(port=8080):
    """Start the dashboard web server"""
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f"🌐 Dashboard started at http://0.0.0.0:{port}")
    print(f"📝 Access the dashboard to monitor progress")
    
    # Start in a separate thread
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server

def get_ip_address():
    """Get the IP address for remote access"""
    try:
        # Get hostname
        hostname = socket.gethostname()
        # Get IP
        ip = socket.gethostbyname(hostname)
        return ip
    except:
        return "localhost"

if __name__ == "__main__":
    import sys
    
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    
    print("="*60)
    print("🚀 GPU Kernel Benchmark Monitor")
    print("="*60)
    print(f"📊 Dashboard: http://{get_ip_address()}:{port}")
    print(f"📁 State file: {state.state_file.absolute()}")
    print(f"📄 Log file: {state.log_file.absolute()}")
    print("="*60)
    
    server = start_dashboard(port)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()
