from django.shortcuts import render
from .ml_model import sentiment_predictor
from .speech import speech_to_text
from .ml_model import fetch_books
from django.http import HttpResponse
from django.http import JsonResponse
from .speech import speech_to_text 
from .ml_model import expression_check
from .camera import cam
import json
from .camera import encode_image_to_base64


def home(request):  
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def recommendation(request):
    if request.method == 'GET':
        user_input = request.GET.get('text_input', '')
        if 'text_input' in request.GET:
            user_input = request.GET.get('text_input', '')
        else:
            # If no text input provided, use speech recognition
            user_input = speech_to_text()
        if not user_input:
            return HttpResponse("Error: Unable to get user input")    

        result = sentiment_predictor([user_input])
        mood_to_category = {
            "Surprise": "comedy",
            "Love": "romance",
            "Joy": "adventure",
            "Fear": "horror",
            "Anger": "motivational",
            "Sadness": "inspirational"
        }
        mood = result.get('mood', '').capitalize()  # Get the recognized mood and capitalize it
        book_category = mood_to_category.get(mood, '')  # Get the corresponding book category

        books = None
        if book_category:
            books = fetch_books(book_category)
        return render(request, "recommendation.html", {'result': result, 'books': books})
    

def cam_view(request):
    if request.method == "POST": 
        
        image_data = request.FILES.get('image')
        print("image_data:",image_data)
        if image_data:
            # You can directly pass image_data to your cam function for processing
            image_64 = encode_image_to_base64(image_data)
            emotion_index = cam(image_64)
            print(emotion_index)  # Assuming cam function accepts image data as bytes
            if emotion_index is not None:
                mood, video_urls = expression_check(emotion_index)
                return render(request, "recommendation.html", {'result': mood, 'videos': video_urls})
            else:
                return JsonResponse({'error': 'Emotion detection failed'}, status=500)
        else:
            return JsonResponse({'error': 'Image not found in request'}, status=400)
    else:
        return JsonResponse({'error': 'Invalid request method'}, status=405)
        