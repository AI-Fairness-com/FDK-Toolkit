# run_tests.py
#!/usr/bin/env python3
"""
Test runner for FDK Justice Pipeline tests
"""

import pytest
import sys
import os

def main():
    """Run all tests"""
    # Add tests directory to path
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, tests_dir)
    
    # Run tests
    print("Running FDK Justice Pipeline Tests...")
    print("=" * 50)
    
    result = pytest.main([
        '-v',           # Verbose output
        '--tb=short',   # Short tracebacks
        'tests/'        # Test directory
    ])
    
    print("=" * 50)
    if result == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    
    sys.exit(result)

if __name__ == '__main__':
    main()
