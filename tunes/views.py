from django.shortcuts import render
from .ml_model import sentiment_predictor
from django.http import HttpResponse

def home(request):  
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def recommendation(request):
    if request.method == 'GET':
        user_input = request.GET.get('text_input', '')
        if not user_input:
            # If no text input provided, handle accordingly (for example, show an error message)
            return HttpResponse("No input provided")
        
        result = sentiment_predictor([user_input])
        return render(request, "recommendation.html", {'result': result})
