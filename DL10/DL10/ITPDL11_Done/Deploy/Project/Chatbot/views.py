from django.shortcuts import render
from django.http import JsonResponse
# import random
# import json
import numpy as np
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
#from .models import Response, models
from .processor import chatbot_response
# Remove the comments to download additional nltk packages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST


# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# from sklearn.feature_extraction.text import TfidfVectorizer

# from tensorflow import keras
# import tensorflow as tf 
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout

def index(request):
    return render(request,"app/chatbot.html")


@require_POST
@csrf_exempt
def chatbot_response_view(request):
    if request.method == 'POST':
        the_question = request.POST.get('question', '')

        response = chatbot_response(the_question)
        print(response)

        return JsonResponse({"response": response})
    else:
        
        return JsonResponse({"message": "This endpoint only accepts POST requests."})
