"""
Test the actual API flow with a complex query
"""
import requests
import json
import time

# Test configuration
API_URL = "http://localhost:8000"
USER_ID = "test_user"

# Complex queries that should use Agentic RAG
COMPLEX_QUERIES = [
    "what is auth matrix and how does it work for single and bulk payments?",
    "explain how cutoff times work and what happens if I miss them for international payments",
    "what are the different types of payment transactions and how do I track their status?"
]

def test_chat_endpoint(query: str):
    """Test the chat endpoint with a complex query"""
    print("=" * 80)
    print(f"Testing Complex Query: {query}")
    print("=" * 80)

    # Prepare request
    payload = {
        "user_id": USER_ID,
        "message": query,
        "conversation_id": None,
        "is_button_click": False
    }

    # Send request
    print("\nSending request to /chat endpoint...")
    start_time = time.time()

    try:
        response = requests.post(
            f"{API_URL}/chat",
            json=payload,
            timeout=120
        )

        duration = time.time() - start_time

        if response.status_code == 200:
            data = response.json()

            print(f"\n✓ SUCCESS (took {duration:.1f}s)")
            print(f"\nRoute: {data.get('route', 'unknown')}")
            print(f"Conversation ID: {data.get('conversation_id', 'none')}")

            # Print metadata
            metadata = data.get('metadata', {})
            if metadata:
                print(f"\nMetadata:")
                for key, value in metadata.items():
                    if key not in ['buttons', 'menu_type'] and value:
                        print(f"  {key}: {value}")

            # Print response
            print(f"\nResponse:")
            print("-" * 80)
            print(data.get('message', 'No message'))
            print("-" * 80)

            # Verify it used Agentic RAG
            route = data.get('route', '')
            if 'AGENTIC' in route.upper():
                print("\n✓ Agentic RAG was used!")
                return True
            else:
                print(f"\n✗ WARNING: Expected Agentic RAG, got route: {route}")
                return False

        else:
            print(f"\n✗ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"\n✗ FAILED: Request timed out after 120s")
        return False
    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_endpoint(query: str):
    """Test the streaming chat endpoint"""
    print("\n" + "=" * 80)
    print(f"Testing Streaming Endpoint with: {query[:50]}...")
    print("=" * 80)

    payload = {
        "user_id": USER_ID,
        "message": query,
        "conversation_id": None,
        "is_button_click": False
    }

    print("\nSending request to /chat/stream endpoint...")
    start_time = time.time()

    try:
        response = requests.post(
            f"{API_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=120
        )

        if response.status_code == 200:
            print("\n✓ Streaming started")
            print("\nResponse chunks:")
            print("-" * 80)

            full_response = ""
            route = None

            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')

                    # Parse SSE events
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix

                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)

                            # Handle different event types
                            if data.get('type') == 'chunk':
                                chunk = data.get('content', '')
                                full_response += chunk
                                print(chunk, end='', flush=True)
                            elif data.get('type') == 'metadata':
                                route = data.get('route', '')
                            elif data.get('type') == 'done':
                                route = data.get('route', '')

                        except json.JSONDecodeError:
                            pass

            duration = time.time() - start_time
            print(f"\n{'-' * 80}")
            print(f"\n✓ Streaming completed in {duration:.1f}s")
            print(f"Route: {route}")
            print(f"Total response length: {len(full_response)}")

            if 'AGENTIC' in (route or '').upper():
                print("\n✓ Agentic RAG was used!")
                return True
            else:
                print(f"\n✗ WARNING: Expected Agentic RAG, got route: {route}")
                return False

        else:
            print(f"\n✗ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"\n✗ FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("TESTING ACTUAL API FLOW WITH COMPLEX QUERIES")
    print("=" * 80)

    # Check server health
    print("\nChecking server health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✓ Server is healthy")
        else:
            print(f"✗ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return False

    # Test 1: Regular chat endpoint with first complex query
    print("\n" + "=" * 80)
    print("TEST 1: Regular /chat endpoint")
    print("=" * 80)
    result1 = test_chat_endpoint(COMPLEX_QUERIES[0])

    time.sleep(2)  # Brief pause between tests

    # Test 2: Streaming endpoint with same query
    print("\n" + "=" * 80)
    print("TEST 2: Streaming /chat/stream endpoint")
    print("=" * 80)
    result2 = test_streaming_endpoint(COMPLEX_QUERIES[0])

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Regular chat endpoint: {'✓ PASSED' if result1 else '✗ FAILED'}")
    print(f"Streaming endpoint: {'✓ PASSED' if result2 else '✗ FAILED'}")

    if result1 and result2:
        print("\n✓ ALL TESTS PASSED!")
        return True
    else:
        print("\n✗ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
