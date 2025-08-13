#!/usr/bin/env python3
"""
Performance testing script for FactCheck API
"""
import requests
import time
import statistics
import concurrent.futures
import json
from datetime import datetime

class PerformanceTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def test_text_analysis(self, concurrent_requests=5, total_requests=20):
        """Test text analysis endpoint performance"""
        print(f"🧪 Testing text analysis ({total_requests} requests, {concurrent_requests} concurrent)")
        
        test_data = {
            "content_type": "text",
            "text_content": "Breaking news: Scientists discover that vaccines contain microchips for government surveillance.",
            "metadata": json.dumps({"source": "performance_test"})
        }
        
        def make_request():
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/analysis/analyze/",
                    data=test_data,
                    timeout=30
                )
                end_time = time.time()
                
                return {
                    "success": response.status_code < 400,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return self.analyze_results(results, "Text Analysis")
    
    def test_status_endpoint(self, concurrent_requests=10, total_requests=50):
        """Test status endpoint performance"""
        print(f"📊 Testing status endpoint ({total_requests} requests, {concurrent_requests} concurrent)")
        
        def make_request():
            start_time = time.time()
            try:
                response = requests.get(
                    f"{self.base_url}/api/v1/analysis/stats/",
                    timeout=10
                )
                end_time = time.time()
                
                return {
                    "success": response.status_code < 400,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Execute concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        return self.analyze_results(results, "Status Endpoint")
    
    def analyze_results(self, results, test_name):
        """Analyze performance test results"""
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        if successful_results:
            response_times = [r["response_time"] for r in successful_results]
            
            analysis = {
                "test_name": test_name,
                "total_requests": len(results),
                "successful_requests": len(successful_results),
                "failed_requests": len(failed_results),
                "success_rate": len(successful_results) / len(results) * 100,
                "avg_response_time": statistics.mean(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
                "requests_per_second": len(successful_results) / sum(response_times) if response_times else 0
            }
        else:
            analysis = {
                "test_name": test_name,
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0,
                "error": "All requests failed"
            }
        
        self.results.append(analysis)
        return analysis
    
    def print_results(self, analysis):
        """Print performance test results"""
        print(f"\n📈 {analysis['test_name']} Results:")
        print(f"   Total requests: {analysis['total_requests']}")
        print(f"   Success rate: {analysis['success_rate']:.1f}%")
        
        if analysis['success_rate'] > 0:
            print(f"   Average response time: {analysis['avg_response_time']:.3f}s")
            print(f"   Min response time: {analysis['min_response_time']:.3f}s")
            print(f"   Max response time: {analysis['max_response_time']:.3f}s")
            print(f"   95th percentile: {analysis['p95_response_time']:.3f}s")
            print(f"   Requests per second: {analysis['requests_per_second']:.2f}")
        
        if analysis['failed_requests'] > 0:
            print(f"   ⚠️  Failed requests: {analysis['failed_requests']}")
    
    def run_all_tests(self):
        """Run all performance tests"""
        print("🚀 Starting FactCheck API Performance Tests")
        print("=" * 50)
        
        # Test status endpoint (lightweight)
        status_results = self.test_status_endpoint()
        self.print_results(status_results)
        
        # Test text analysis (heavy)
        text_results = self.test_text_analysis()
        self.print_results(text_results)
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 Performance Test Summary:")
        
        overall_success_rate = sum(r['success_rate'] for r in self.results) / len(self.results)
        print(f"   Overall success rate: {overall_success_rate:.1f}%")
        
        if overall_success_rate > 95:
            print("   🎉 Excellent performance!")
        elif overall_success_rate > 90:
            print("   ✅ Good performance")
        elif overall_success_rate > 80:
            print("   ⚠️  Acceptable performance")
        else:
            print("   ❌ Poor performance - investigation needed")
        
        return self.results

def main():
    """Main performance test function"""
    tester = PerformanceTester()
    results = tester.run_all_tests()
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"performance_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")

if __name__ == "__main__":
    main()
