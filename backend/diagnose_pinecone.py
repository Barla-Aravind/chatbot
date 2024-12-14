import os
import sys
import requests
import platform
import socket
import math

def calculate_entropy(key):
    """
    Calculate the entropy of a string to assess its randomness
    """
    char_counts = {}
    for char in key:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    entropy = 0
    for count in char_counts.values():
        prob = count / len(key)
        entropy -= prob * math.log2(prob)  # Corrected entropy calculation
    
    return entropy

def network_diagnostics():
    print("\n🌐 Network Diagnostics")
    print("=====================")
    
    # DNS Resolution
    try:
        socket.gethostbyname('api.pinecone.io')
        print("✅ DNS Resolution: Successful")
    except socket.gaierror:
        print("❌ DNS Resolution Failed for api.pinecone.io")
    
    # Network Connectivity Tests
    test_urls = [
        'https://api.pinecone.io',
        'https://www.pinecone.io',
        'https://pinecone.io'
    ]
    
    for url in test_urls:
        try:
            response = requests.get(url, timeout=5)
            print(f"✅ Connection to {url}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection to {url} failed: {e}")

def detailed_key_analysis(api_key):
    print("\n🔍 Detailed API Key Analysis")
    print("===========================")
    
    # Key Structure Analysis
    print(f"Key Length: {len(api_key)} characters")
    print(f"Key Prefix: {api_key[:6]}")
    
    # Entropy and Randomness Check
    try:
        key_entropy = calculate_entropy(api_key)
        print(f"Key Entropy: {key_entropy:.2f}")
    except Exception as e:
        print(f"❌ Entropy Calculation Error: {e}")
    
    # Pattern and Complexity Check
    import re
    complexity_checks = {
        "Contains Lowercase": bool(re.search(r'[a-z]', api_key)),
        "Contains Uppercase": bool(re.search(r'[A-Z]', api_key)),
        "Contains Numbers": bool(re.search(r'\d', api_key)),
        "Contains Special Chars": bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', api_key))
    }
    
    print("\nKey Complexity:")
    for check, result in complexity_checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")

def validate_pinecone_api_key(api_key):
    print("\n🔍 Pinecone API Key Validation")
    print("==============================")

    # Detailed Key Format Checks
    key_patterns = {
        'csk_': "Standard Pinecone API Key",
        'pcsk_': "Project-specific Key",
    }

    matching_pattern = None
    for prefix, description in key_patterns.items():
        if api_key.startswith(prefix):
            matching_pattern = prefix
            print(f"🔍 Detected Key Type: {description}")
            break

    if not matching_pattern:
        print("❌ Unrecognized API key format")
        print("Known formats: csk_, pcsk_")
        return False

    # Length and complexity check
    if len(api_key) < 30:
        print("❌ API key seems unusually short")
        return False

    # Network-based validation
    try:
        print("🔍 Attempting to validate API key with Pinecone...")
        
        headers = {
            'Api-Key': api_key,
            'Accept': 'application/json'
        }
        
        response = requests.get(
            'https://api.pinecone.io/control/list-indexes', 
            headers=headers,
            timeout=10
        )
        
        print(f"\nHTTP Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API Key is valid!")
            try:
                indexes = response.json()
                print("\nExisting Indexes:")
                for index in indexes:
                    print(f"- {index.get('name', 'Unnamed Index')}")
            except Exception as json_err:
                print(f"Note: Could not parse index details: {json_err}")
            return True
        
        elif response.status_code == 401:
            print("❌ Authentication Failed")
            print("Possible reasons:")
            print("1. Incorrect API Key")
            print("2. Revoked or Expired Key")
            print("3. Account Access Issues")
            print("\nResponse Details:")
            print(response.text)
            return False
        
        else:
            print(f"❓ Unexpected response: {response.status_code}")
            print(response.text)
            return False
    
    except requests.exceptions.RequestException as req_err:
        print(f"❌ Network Error: {req_err}")
        print("Possible issues:")
        print("1. Internet connectivity")
        print("2. Firewall blocking Pinecone API")
        print("3. Pinecone service down")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def comprehensive_pinecone_diagnostic():
    print("Comprehensive Pinecone Diagnostic")
    print("=================================")

    # System Information
    print("\n🔍 System Information:")
    print(f"Python Version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Hostname: {platform.node()}")

    # Library Check
    print("\n🔍 Pinecone Library Check:")
    try:
        import pinecone
        print(f"Pinecone Version: {pinecone.__version__}")
    except Exception as import_err:
        print(f"❌ Pinecone import failed: {import_err}")
        sys.exit(1)

    # API Key Validation
    print("\n🔍 API Key Validation:")
    api_key = os.environ.get('PINECONE_API_KEY')
    
    if not api_key:
        print("❌ PINECONE_API_KEY environment variable not set!")
        sys.exit(1)

    # Mask API key for security
    masked_key = f"{api_key[:5]}{'*' * (len(api_key) - 10)}{api_key[-5:]}"
    print(f"API Key (masked): {masked_key}")

    # Detailed Key Analysis
    detailed_key_analysis(api_key)

    # Pinecone-Specific API Validation
    validation_result = validate_pinecone_api_key(api_key)

    # Network Diagnostics
    network_diagnostics()

    if not validation_result:
        print("\n🚨 Recommended Actions:")
        print("1. Verify Pinecone Dashboard API Key")
        print("2. Regenerate API Key")
        print("3. Check Pinecone Account Status")
        print("4. Contact Pinecone Support if issues persist")
        sys.exit(1)

    print("\n✅ All Diagnostics Completed Successfully!")

if __name__ == "__main__":
    comprehensive_pinecone_diagnostic()