#!/usr/bin/env python3
"""
Test script to verify proxy configuration
"""

import os
import urllib.request
import urllib.error

def test_proxy():
    """Test if proxy is working"""
    print("Testing proxy configuration...")
    print(f"HTTP_PROXY: {os.environ.get('HTTP_PROXY', 'Not set')}")
    print(f"HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")

    # Set up proxy handler
    # proxy_handler = urllib.request.ProxyHandler({
    #     'http': os.environ.get('HTTP_PROXY'),
    #     'https': os.environ.get('HTTPS_PROXY')
    # })
    # opener = urllib.request.build_opener(proxy_handler)

    # try:
    #     # Test connection to a simple HTTP endpoint
    #     response = opener.open('http://httpbin.org/ip', timeout=10)
    #     data = response.read().decode('utf-8')
    #     print(f"✅ Proxy test successful: {data}")
    #     return True
    # except (urllib.error.URLError, OSError) as e:
    #     print(f"❌ Proxy test failed: {e}")
    #     return False

if __name__ == "__main__":
    # Set proxy if not already set
    if 'HTTP_PROXY' not in os.environ:
        os.environ['HTTP_PROXY'] = 'http://127.0.0.1:1087'
        os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:1087'
        print("Proxy environment variables set")

    # success = test_proxy()
    # if success:
    #     print("\n🎉 Proxy is working correctly!")
    # else:
    #     print("\n⚠️  Proxy may not be working. Please check:")
    #     print("1. Is your proxy server running on 127.0.0.1:1087?")
    #     print("2. Is the proxy server configured correctly?")
    #     print("3. Try disabling proxy by commenting out the proxy lines in the notebook")
