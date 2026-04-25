#!/usr/bin/env python3
"""
Test script to demonstrate the API documentation features of the Hunyuan3D API server.

This script shows how to use the API endpoints with proper parameter documentation.
"""

import requests
import base64
import json
from PIL import Image
import io
import time
import os

# API server URL (adjust as needed)
API_BASE_URL = "http://localhost:8081"

def load_demo_image():
    """Load the demo image from assets/demo.png"""
    try:
        # Load the demo image
        image_path = 'assets/demo.png'
        image = Image.open(image_path).convert("RGBA")
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        print(f"Loaded demo image from: {image_path}")
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
        
        return img_base64
    except FileNotFoundError:
        print(f"Error: Demo image not found at {image_path}")
        print("Creating fallback test image...")
        return create_test_image()
    except Exception as e:
        print(f"Error loading demo image: {e}")
        print("Creating fallback test image...")
        return create_test_image()

def create_test_image():
    """Create a simple test image for API testing (fallback)"""
    # Create a simple 256x256 test image
    img = Image.new('RGB', (256, 256), color='red')
    
    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return img_base64

def save_glb_file(response, filename):
    """Save GLB file from response content"""
    try:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"GLB file saved as: {filename}")
        return True
    except Exception as e:
        print(f"Error saving GLB file: {e}")
        return False

def save_base64_glb(base64_data, filename):
    """Save GLB file from base64 encoded data"""
    try:
        # Decode base64 data
        glb_data = base64.b64decode(base64_data)
        
        # Save to file
        with open(filename, 'wb') as f:
            f.write(glb_data)
        print(f"GLB file saved as: {filename}")
        return True
    except Exception as e:
        print(f"Error saving GLB file from base64: {e}")
        return False

def test_generation_request():
    """Test the generation request with simplified parameters"""
    print("Loading demo image...")
    # Load demo image
    demo_image = load_demo_image()
    
    # Simplified request payload with only the parameters the worker actually uses
    request_data = {
        "image": demo_image,
        "type": "glb"
    }
    
    print("Testing /generate endpoint...")
    print("Request parameters:")
    for key, value in request_data.items():
        if key == "image":
            print(f"  {key}: [base64 encoded demo image data]")
        else:
            print(f"  {key}: {value}")
    
    try:
        response = requests.post(f"{API_BASE_URL}/generate", json=request_data)
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            print("Success! Generated 3D model file received.")
            
            # Save the GLB file
            timestamp = int(time.time())
            filename = f"generated_model_{timestamp}.glb"
            if save_glb_file(response, filename):
                print(f"Model saved successfully to: {filename}")
            else:
                print("Failed to save model file")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server. Make sure it's running on localhost:8081")

def test_async_generation():
    """Test the asynchronous generation endpoint"""
    
    demo_image = load_demo_image()
    
    request_data = {
        "image": demo_image,
        "type": "glb"
    }
    
    print("\nTesting /send endpoint (async)...")
    try:
        response = requests.post(f"{API_BASE_URL}/send", json=request_data)
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            uid = result.get("uid")
            print(f"Task ID: {uid}")
            
            # Check status
            print("Checking task status...")
            status_response = requests.get(f"{API_BASE_URL}/status/{uid}")
            print(f"Status: {status_response.json()}")
            
            # Poll status until completed
            while True:
                status_response = requests.get(f"{API_BASE_URL}/status/{uid}")
                status_data = status_response.json()
                print(f"Status: {status_data['status']}")
                
                if status_data['status'] == 'completed':
                    print("Generation completed!")
                    
                    # Save the GLB file from base64 data
                    model_base64 = status_data.get('model_base64')
                    if model_base64:
                        timestamp = int(time.time())
                        filename = f"async_generated_model_{uid}_{timestamp}.glb"
                        if save_base64_glb(model_base64, filename):
                            print(f"Model saved successfully to: {filename}")
                        else:
                            print("Failed to save model file")
                    else:
                        print("No model data received in response")
                    break
                elif status_data['status'] == 'error':
                    print(f"Error: {status_data.get('message', 'Unknown error')}")
                    break
                    
                time.sleep(2)  # Wait 2 seconds between checks
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server.")

def test_health_check():
    """Test the health check endpoint"""
    
    print("\nTesting /health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Health: {result}")
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("Could not connect to API server.")

def show_api_documentation_info():
    """Show information about the API documentation"""
    
    print("=" * 60)
    print("HUNYUAN3D API DOCUMENTATION")
    print("=" * 60)
    print()
    print("The API server now includes comprehensive documentation:")
    print()
    print("1. Pydantic Models:")
    print("   - GenerationRequest: Documents all input parameters")
    print("   - GenerationResponse: Documents response format")
    print("   - StatusResponse: Documents status endpoint response")
    print("   - HealthResponse: Documents health check response")
    print()
    print("2. Parameter Documentation:")
    print("   - All parameters have descriptions and examples")
    print("   - Parameter types and constraints are defined")
    print("   - Default values are specified")
    print("   - Note: Only 'image' and 'type' parameters are currently used")
    print()
    print("3. API Organization:")
    print("   - Endpoints are tagged (generation, status)")
    print("   - Comprehensive descriptions for each endpoint")
    print("   - Example requests and responses")
    print()
    print("4. Access Documentation:")
    print(f"   - Interactive docs: {API_BASE_URL}/docs")
    print(f"   - Alternative docs: {API_BASE_URL}/redoc")
    print()
    print("5. Available Endpoints:")
    print("   - POST /generate - Immediate 3D generation")
    print("   - POST /send - Async 3D generation")
    print("   - GET /status/{uid} - Check task status")
    print("   - GET /health - Service health check")
    print()
    print("6. Simplified Parameters:")
    print("   - image: Base64 encoded input image (required)")
    print("   - type: Output file format - 'glb' or 'obj' (optional, default: 'glb')")
    print()
    print("7. File Saving:")
    print("   - GLB files are automatically saved with timestamps")
    print("   - Direct generation saves as: generated_model_{timestamp}.glb")
    print("   - Async generation saves as: async_generated_model_{uid}_{timestamp}.glb")
    print()

if __name__ == "__main__":
    show_api_documentation_info()
    
    # Run tests if server is available
    test_health_check()
    test_generation_request()
    test_async_generation()
    
    print("\n" + "=" * 60)
    print("To view the interactive API documentation:")
    print(f"1. Start the API server: python api_server.py")
    print(f"2. Open your browser to: {API_BASE_URL}/docs")
    print("3. Explore the documented endpoints and parameters")
    print("4. Generated GLB files will be saved in the current directory")
    print("=" * 60) 