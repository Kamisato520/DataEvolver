# Hunyuan3D API Documentation

This document explains how the FastAPI documentation has been enhanced to provide comprehensive parameter documentation for the Hunyuan3D API server.

## Overview

The API server now uses Pydantic models to automatically generate interactive documentation that includes:

- **Parameter descriptions and types**
- **Default values and constraints**
- **Example requests and responses**
- **Organized endpoint groups**
- **Interactive testing interface**

## Key Improvements

### 1. Pydantic Models

The API now uses structured Pydantic models instead of raw JSON requests:

```python
class GenerationRequest(BaseModel):
    """Request model for 3D generation API"""
    image: str = Field(
        ..., 
        description="Base64 encoded input image for 3D generation",
        example="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    texture: bool = Field(
        False,
        description="Whether to generate textures for the 3D model"
    )
    seed: int = Field(
        1234,
        description="Random seed for reproducible generation",
        ge=0,
        le=2**32-1
    )
    # ... more parameters
```

### 2. Parameter Documentation

Each parameter includes:
- **Description**: What the parameter does
- **Type**: Data type (str, int, float, bool, etc.)
- **Constraints**: Min/max values, allowed values
- **Default values**: What happens if not provided
- **Examples**: Sample values for testing

### 3. API Organization

Endpoints are organized into logical groups using tags:

- **`generation`**: 3D model generation endpoints
- **`status`**: Task status and health check endpoints

### 4. Comprehensive Metadata

The FastAPI app includes:
- **Title and version**
- **Detailed description**
- **Contact information**
- **License information**
- **Feature overview**

## Available Endpoints

### Generation Endpoints

#### POST `/generate`
Generate a 3D model from an input image.

**Parameters:**
- `image` (required): Base64 encoded input image
- `remove_background` (optional): Auto-remove background (default: true)
- `texture` (optional): Generate textures (default: false)
- `seed` (optional): Random seed (default: 1234)
- `octree_resolution` (optional): Mesh resolution (default: 256)
- `num_inference_steps` (optional): Generation steps (default: 5)
- `guidance_scale` (optional): Generation guidance (default: 5.0)
- `num_chunks` (optional): Processing chunks (default: 8000)
- `face_count` (optional): Max faces for textures (default: 40000)
- `type` (optional): Output format (default: "glb")

#### POST `/send`
Start asynchronous 3D generation task.

**Parameters:** Same as `/generate`
**Returns:** Task ID for status tracking

### Status Endpoints

#### GET `/health`
Check service health status.

#### GET `/status/{uid}`
Check task status and retrieve results.

## Accessing the Documentation

### Interactive Documentation

1. Start the API server:
   ```bash
   python api_server.py
   ```

2. Open your browser to:
   - **Swagger UI**: `http://localhost:8081/docs`
   - **ReDoc**: `http://localhost:8081/redoc`

### Features of the Interactive Docs

- **Try it out**: Test endpoints directly from the browser
- **Parameter validation**: Automatic validation of input parameters
- **Response examples**: See expected response formats
- **Error handling**: Understand possible error responses
- **Authentication**: Configure if needed (currently not required)

## Example Usage

### Basic 3D Generation

```python
import requests
import base64

# Load and encode image
with open("input_image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Prepare request
request_data = {
    "image": image_data,
    "texture": True,
    "seed": 42,
    "type": "glb"
}

# Send request
response = requests.post("http://localhost:8081/generate", json=request_data)

if response.status_code == 200:
    # Save the generated 3D model
    with open("output_model.glb", "wb") as f:
        f.write(response.content)
```

### Asynchronous Generation

```python
# Start async task
response = requests.post("http://localhost:8081/send", json=request_data)
task_id = response.json()["uid"]

# Check status
status_response = requests.get(f"http://localhost:8081/status/{task_id}")
status = status_response.json()

if status["status"] == "completed":
    # Decode and save the model
    model_data = base64.b64decode(status["model_base64"])
    with open("async_model.glb", "wb") as f:
        f.write(model_data)
```

## Testing

Use the provided test script to verify the API:

```bash
python test_api_docs.py
```

This script demonstrates:
- Parameter validation
- Request formatting
- Response handling
- Error scenarios

## Benefits

### For Developers
- **Self-documenting API**: Parameters are clearly defined
- **Type safety**: Automatic validation prevents errors
- **Interactive testing**: Try endpoints without writing code
- **Clear examples**: See exactly what to send and expect

### For Users
- **Easy integration**: Clear parameter documentation
- **Error prevention**: Validation catches issues early
- **Quick testing**: Interactive interface for exploration
- **Comprehensive examples**: Working code samples

## Technical Details

### Dependencies
- `fastapi`: Web framework with automatic documentation
- `pydantic`: Data validation and serialization
- `uvicorn`: ASGI server

### File Structure
```
api_server.py          # Main API server with Pydantic models
test_api_docs.py       # Test script demonstrating usage
API_DOCUMENTATION.md   # This documentation file
```

### Customization

To add new parameters or endpoints:

1. **Add to Pydantic model**:
   ```python
   new_param: str = Field(
       "default_value",
       description="Parameter description"
   )
   ```

2. **Update endpoint function**:
   ```python
   @app.post("/new_endpoint", tags=["category"])
   async def new_endpoint(request: RequestModel):
       """Endpoint description"""
       # Implementation
   ```

3. **Documentation updates automatically**!

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Port conflicts**: Change port in `api_server.py` if needed
3. **Model loading**: Check model paths and GPU availability

### Getting Help

- Check the interactive documentation at `/docs`
- Review the test script for working examples
- Examine the Pydantic models for parameter details

## Conclusion

The enhanced API documentation provides a professional, user-friendly interface for the Hunyuan3D API. Users can now understand all parameters, test endpoints interactively, and integrate the API more easily into their applications. 