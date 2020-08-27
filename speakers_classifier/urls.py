from django.urls import path
import speakers_classifier.views as views

urlpatterns = [
    path('predict_gender/', views.predict_gender, name = 'predict_gender')
]