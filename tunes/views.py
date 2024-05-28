from django.shortcuts import render
from .ml_model import sentiment_predictor, cam_sentiment_predictor
# from django.http import HttpResponse, JsonResponse
# from .camera import cam
import json
from .camera import encode_image_to_base64, cam
# from django.http import JsonResponse

def home(request):  
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def recommendation(request):
    if request.method == 'GET':
        user_input = request.GET.get('text_input', '')
        if not user_input:
            return render(request, "notFound.html")
        
        result = sentiment_predictor([user_input])
        return render(request, "recommendation.html", {'result': result})

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
                result = cam_sentiment_predictor(emotion_index)
                # mood, video_urls = expression_check(emotion_index)
                # print("MOOD:", mood)
                # print(video_urls)
                # return HttpResponse({'mood': mood, 'video_urls': video_urls})
                return render(request, "recommendation.html", {'result': result})
            else:
                # return JsonResponse({'error': 'Emotion detection failed'}, status=500)
                return render(request, "notFound.html")
        else:
            # return JsonResponse({'error': 'Image not found in request'}, status=400)
            return render(request, "notFound.html")
    else:
        # return JsonResponse({'error': 'Invalid request method'}, status=405)
        return render(request, "notFound.html")