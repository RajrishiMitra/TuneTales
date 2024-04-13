from django.shortcuts import render
from .ml_model import sentiment_predictor
from .speech import speech_to_text
from .ml_model import fetch_books

from django.http import HttpResponse
def home(request):  
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def recommendation(request):
    if request.method == 'GET':
        user_input = request.GET.get('text_input', '')
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
