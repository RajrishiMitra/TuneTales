from django.shortcuts import render,redirect
from .ml_model import sentiment_predictor, cam_sentiment_predictor
from .camera import encode_image_to_base64, cam

def home(request):  
    return render(request, "home.html")

def about(request):
    return render(request, "about.html")

def recommendation(request):
    if request.method == 'GET':
        result = request.session.get('result')
        user_input = request.GET.get('text_input', '')
        if not user_input:
            return render(request, "notFound.html")
        
        result = sentiment_predictor([user_input])
        return render(request, "recommendation.html", {'result': result})

def cam_view(request):
    if request.method == "POST": 
        image_data = request.FILES.get('image')
        print(image_data.size)
        print("image_data:",image_data)
        if image_data:
            image_64 = encode_image_to_base64(image_data)
            emotion_index = cam(image_64)
            print("Emotion Index:",emotion_index)
            if emotion_index is not None:
                result = cam_sentiment_predictor(emotion_index)
                request.session['result'] = result
                return render(request,'recommendation.html', {'result': result})
            else:
                # return JsonResponse({'error': 'Emotion detection failed'}, status=500)
                return render(request, "notFound.html")
        else:
            # return JsonResponse({'error': 'Image not found in request'}, status=400)
            return render(request, "notFound.html")
    else:
        # return JsonResponse({'error': 'Invalid request method'}, status=405)
        return render(request, "notFound.html")
    



