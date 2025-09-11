import requests
import json
from pprint import pprint

# Base URL for your API
BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, params=None):
    """Test an API endpoint and print results"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        response = requests.get(url, params=params)
        print(f"\n🔍 Testing: {endpoint}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS!")
            
            # Pretty print first few results if it's a list
            if isinstance(data, list) and len(data) > 0:
                print(f"Found {len(data)} results. First result:")
                pprint(data[0])
            else:
                pprint(data)
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Make sure your API is running!")
        print("Run: uvicorn main:app --reload")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def main():
    """Run all API tests"""
    print("🚀 Testing Campus Data API")
    print("=" * 50)
    
    # Test 1: Health check
    test_endpoint("/health")
    
    # Test 2: Root endpoint
    test_endpoint("/")
    
    # Test 3: Get students with filters
    test_endpoint("/students", {"limit": 3, "active_only": True})
    
    # Test 4: Get students by major
    test_endpoint("/students", {
        "major": "Computer Science", 
        "limit": 5,
        "min_gpa": 3.0
    })
    
    # Test 5: At-risk students
    test_endpoint("/students/at-risk", {"gpa_threshold": 2.5})
    
    # Test 6: Student analytics
    test_endpoint("/analytics/students")
    
    # Test 7: Course analytics
    test_endpoint("/analytics/courses", {"department": "CS"})
    
    # Test 8: Get specific student (you might need to change this ID)
    test_endpoint("/students/STU000001")
    
    # Test 9: ML prediction
    test_endpoint("/ml/predictions/student-success/STU000001")
    
    print("\n🎉 All tests completed!")
    print("\n💡 Pro tip: Visit http://localhost:8000/docs for interactive testing!")

if __name__ == "__main__":
    main()
