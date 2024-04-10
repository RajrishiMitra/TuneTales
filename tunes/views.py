from django.shortcuts import render
from django.http import HttpResponse
from .ml_model import sentiment_predictor


# Create your views here.

def home(request):  
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def recommendation(request):
    if request.method == 'GET':
        user_input = request.GET.get('text_input', '')
        result = sentiment_predictor([user_input])
        return render(request, "recommendation.html", {'result': result})