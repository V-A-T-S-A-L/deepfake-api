from django.shortcuts import render
import os
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
import firebase_admin
from firebase_admin import credentials, firestore
import json
from functools import wraps

# Initialize Firebase (assuming you have a serviceAccountKey.json file)
try:
    cred = credentials.Certificate("myapp/service_key/service_key.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_INITIALIZED = True
except Exception as e:
    print(f"Firebase initialization error: {str(e)}")
    FIREBASE_INITIALIZED = False

# Load the image detection model
MODEL_PATH = "myapp/models/model.h5"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def api_key_required(view_func):
    """Decorator to check if API key is valid"""
    @wraps(view_func)
    def wrapped_view(request, *args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return JsonResponse({
                "error": "API key is required",
                "status": "unauthorized"
            }, status=401)
            
        if not FIREBASE_INITIALIZED:
            return JsonResponse({
                "error": "Authentication service unavailable",
                "status": "service_unavailable"
            }, status=503)
            
        # Validate API key with Firebase
        try:
            # Query Firestore for the API key
            api_keys_ref = db.collection('api_keys')
            query = api_keys_ref.where('key', '==', api_key).limit(1).get()
            
            if not query or len(query) == 0:
                return JsonResponse({
                    "error": "Invalid API key",
                    "status": "unauthorized"
                }, status=401)
                
            # Check if key is active
            key_data = query[0].to_dict()
            if not key_data.get('active', False):
                return JsonResponse({
                    "error": "API key is inactive",
                    "status": "unauthorized"
                }, status=401)
                
            # Proceed with the view
            return view_func(request, *args, **kwargs)
            
        except Exception as e:
            print(f"API key validation error: {str(e)}")
            return JsonResponse({
                "error": "Authentication error",
                "status": "internal_error"
            }, status=500)
            
    return wrapped_view

def generate_gradcam(image_path, model, last_conv_layer_name="conv2d_2"):
    """Generate Grad-CAM visualization for model explainability"""
    try:
        # Find the target convolutional layer
        for i, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                target_layer = layer
                break
        else:
            # If target layer not found, find the last conv layer
            for i in reversed(range(len(model.layers))):
                if 'conv' in model.layers[i].name.lower():
                    target_layer = model.layers[i]
                    print(f"Using {target_layer.name} instead of {last_conv_layer_name}")
                    break
            else:
                # If no conv layer found, return a simple heatmap
                img = cv2.imread(image_path)
                img = cv2.resize(img, (128, 128))
                _, buffer = cv2.imencode('.jpg', img)
                return base64.b64encode(buffer).decode('utf-8')
        
        # Load and preprocess the image
        img = load_img(image_path, target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Create a model that outputs the activations of the target conv layer
        activation_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=target_layer.output
        )
        
        # Get activations
        activations = activation_model.predict(img_array)
        
        # Create a heatmap from the activations
        heatmap = np.mean(activations[0], axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        
        # Resize and apply color map
        heatmap = cv2.resize(heatmap, (128, 128))
        heatmap = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose the heatmap on the original image
        orig_img = cv2.imread(image_path)
        orig_img = cv2.resize(orig_img, (128, 128))
        superimposed_img = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
        
        # Convert to base64 for web display
        _, buffer = cv2.imencode('.jpg', superimposed_img)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return heatmap_base64
        
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        # Return a blank image or the original image as fallback
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        _, buffer = cv2.imencode('.jpg', img)
        return base64.b64encode(buffer).decode('utf-8')

def detect_face(image_path):
    """Detect if image contains a face"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def classify_image(image_path):
    """Classify image as real or fake"""
    if model is None:
        return "Error", 0, "Model not loaded"
        
    image = load_img(image_path, target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array)
    result = 'Fake' if prediction[0][0] > 0.5 else 'Real'
    confidence = (1 - prediction[0][0]) * 100 if result == 'Real' else prediction[0][0] * 100
    
    face_detectable = detect_face(image_path)
    explanation = ""
    
    if result == 'Real':
        explanation = "The image is real. The facial features are natural and well-aligned with the background."
    else:
        explanation = "The image is fake. Possible inconsistencies detected in texture, edges, or alignment. "
        explanation += "Face detection status: " + ("Face detected" if face_detectable else "No face detected.")
    
    return result, confidence, explanation

@csrf_exempt
@api_key_required
def detect_image(request):
    """API endpoint for image deepfake detection"""
    if request.method != 'POST':
        return JsonResponse({
            "error": "Method not allowed",
            "status": "method_not_allowed"
        }, status=405)
        
    if 'image' not in request.FILES:
        return JsonResponse({
            "error": "No image file provided",
            "status": "bad_request"
        }, status=400)
        
    uploaded_file = request.FILES['image']
    file_name = uploaded_file.name
    file_extension = os.path.splitext(file_name)[1].lower()
    
    # Validate file type
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    if file_extension not in allowed_extensions:
        return JsonResponse({
            "error": "Invalid file type. Supported formats: JPG, JPEG, PNG, BMP",
            "status": "bad_request"
        }, status=400)
        
    # Validate file size (e.g., max 5MB)
    if uploaded_file.size > 5 * 1024 * 1024:
        return JsonResponse({
            "error": "File too large. Maximum size is 5MB",
            "status": "bad_request"
        }, status=400)
        
    try:
        # Save the file temporarily
        file_path = default_storage.save(file_name, ContentFile(uploaded_file.read()))
        file_path_full = os.path.join(default_storage.location, file_path)
        
        # Process image
        result, confidence, explanation = classify_image(file_path_full)
        
        # Generate heatmap for model explainability
        heatmap_b64 = None
        if result != "Error":
            heatmap_b64 = generate_gradcam(file_path_full, model)
        
        # Prepare response
        response_data = {
            "status": "success",
            "result": result,
            "confidence": float(confidence),
            "explanation": explanation
        }
        
        if heatmap_b64:
            response_data["heatmap"] = heatmap_b64
            
        # Clean up temporary file
        if os.path.exists(file_path_full):
            os.remove(file_path_full)
            
        return JsonResponse(response_data)
        
    except Exception as e:
        # Clean up on error
        if 'file_path_full' in locals() and os.path.exists(file_path_full):
            os.remove(file_path_full)
            
        print(f"Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JsonResponse({
            "error": f"Error processing image: {str(e)}",
            "status": "internal_error"
        }, status=500)

@csrf_exempt
def api_health(request):
    """API health check endpoint"""
    health_status = {
        "status": "operational",
        "model_loaded": model is not None,
        "firebase_initialized": FIREBASE_INITIALIZED
    }
    
    # Add component status
    components = []
    
    if model is not None:
        components.append({"name": "model", "status": "operational"})
    else:
        components.append({"name": "model", "status": "unavailable"})
        health_status["status"] = "degraded"
        
    if FIREBASE_INITIALIZED:
        components.append({"name": "firebase", "status": "operational"})
    else:
        components.append({"name": "firebase", "status": "unavailable"})
        health_status["status"] = "degraded"
        
    health_status["components"] = components
    
    return JsonResponse(health_status)