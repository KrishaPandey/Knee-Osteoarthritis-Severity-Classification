from django.urls import path
from . import predict_view  # Import predict_view module

urlpatterns = [
    path('predict/', predict_view.predict_view, name='predict')  # Route for prediction
]
