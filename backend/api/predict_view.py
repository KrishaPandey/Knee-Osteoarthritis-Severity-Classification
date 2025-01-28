import logging
import io
import os
import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
from torchvision import transforms
from PIL import Image, ImageStat
from api.koa_model.complex_cnn import ComplexCNN

# Set up logging
logger = logging.getLogger(__name__)

class KneeXrayPredictor:
    def __init__(self):
        self.model = None
        self.mean = [0.6076]  # Dataset mean
        self.std = [0.1931]   # Dataset std
        self.class_thresholds = [0.60, 0.45, 0.60, 0.70, 0.65]
        self.initialize_model()
        self.setup_preprocessing()

    def initialize_model(self):
        """Initialize and load the model"""
        try:
            model_path = 'api/koa_model/best_model.pth'
            self.model = ComplexCNN(num_classes=5)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def apply_clahe(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        try:
            # Convert PIL image to numpy array
            image_array = np.array(image)
            # Create CLAHE object with specified parameters
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            # Apply CLAHE
            enhanced_image = clahe.apply(image_array)
            # Convert back to PIL image
            return Image.fromarray(enhanced_image)
        except Exception as e:
            logger.error(f"Error applying CLAHE: {e}")
            return image  # Return original image if CLAHE fails

    def setup_preprocessing(self):
        """Set up image preprocessing pipeline"""
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(self.apply_clahe),  # Apply CLAHE using our dedicated function
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def is_valid_xray(self, image):
        """Validate if the image is likely a knee X-ray"""
        try:
            # Check image dimensions
            width, height = image.size
            if width < 100 or height < 100:
                return False  # Too small to be a valid knee X-ray

            # Check aspect ratio
            aspect_ratio = width / height
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                return False  # Unlikely to be a knee X-ray

            # Check grayscale intensity distribution
            stat = ImageStat.Stat(image)
            mean_intensity = stat.mean[0]
            if mean_intensity < 50 or mean_intensity > 200:
                return False  # Too dark or too bright for an X-ray

            # Check variance
            variance = stat.var[0]
            if variance < 50:
                return False  # Too uniform to be an X-ray

            return True
        except Exception as e:
            logger.error(f"Error in X-ray validation: {e}")
            return False

    def predict(self, image_tensor):
        """Make prediction on the input image tensor"""
        try:
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze().tolist()

                predicted_class = torch.argmax(output, dim=1).item()
                max_prob = probabilities[predicted_class]

                if max_prob < self.class_thresholds[predicted_class]:
                    return {
                        "message": f"The model is not confident in predicting Class.",
                        "probabilities": probabilities,
                    }

                return {
                    "predicted_class": predicted_class,
                    "probabilities": probabilities
                }
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return {"message": "Error during prediction."}

# Create a global instance of the predictor
predictor = KneeXrayPredictor()

@csrf_exempt
def predict_view(request):
    """Handle prediction request"""
    if request.method == 'POST':
        try:
            if 'file' not in request.FILES:
                return JsonResponse({"error": "No file provided."}, status=400)

            # Load and validate image
            image_file = request.FILES['file']
            image = Image.open(io.BytesIO(image_file.read())).convert("L")

            if not predictor.is_valid_xray(image):
                return JsonResponse(
                    {"error": "Uploaded image is not a valid knee X-ray."},
                    status=400
                )

            # Preprocess and predict
            image_tensor = predictor.preprocess(image).unsqueeze(0)
            prediction = predictor.predict(image_tensor)

            # Handle error case
            if "message" in prediction:
                return JsonResponse(
                    {
                        "error": prediction["message"],
                        "probabilities": prediction.get("probabilities", [])
                    },
                    status=400,
                )

            # Define grade descriptions
            grade_descriptions = {
                0: "Grade 0 (Normal)",
                1: "Grade 1 (Doubtful)",
                2: "Grade 2 (Minimal)",
                3: "Grade 3 (Moderate)",
                4: "Grade 4 (Severe)"
            }

            # Return formatted response
            return JsonResponse({
                "prediction": f"Predicted Result: {grade_descriptions[prediction['predicted_class']]}",
                "probabilities": [
                    {"class": i, "probability": f"{prob * 100:.2f}%"}
                    for i, prob in enumerate(prediction["probabilities"])
                ],
            })

        except Exception as e:
            logger.error(f"Error in prediction view: {e}")
            return JsonResponse({"error": "Internal server error"}, status=500)
