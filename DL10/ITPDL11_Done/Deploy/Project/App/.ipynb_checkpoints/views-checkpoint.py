from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from .forms import UserImageForm
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from django.contrib.auth.decorators import login_required 
from django.contrib.auth import logout as auth_logout
import numpy as np
import joblib
from .forms import RegisterForm, LoginForm, UpdateUserForm, UpdateProfileForm
from django.contrib.auth import authenticate,login,logout
from .models import UserImageModel
import numpy as np
from tensorflow import keras
from PIL import Image,ImageOps

import pyttsx3
import time


def home(request):
    return render(request, 'users/home.html')

@login_required(login_url='users-register')


def index(request):
    return render(request, 'app/index.html')

class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        # will redirect to the home page if a user tries to access the register page while logged in
        if request.user.is_authenticated:
            return redirect(to='/')

        # else process dispatch as it otherwise normally would
        return super(RegisterView, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            form.save()

            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}')

            return redirect(to='login')

        return render(request, self.template_name, {'form': form})


# Class based view that extends from the built in login view to add a remember me functionality

class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            # set session expiry to 0 seconds. So it will automatically close the session after the browser is closed.
            self.request.session.set_expiry(0)

            # Set session as modified to force data updates/cookie to be saved.
            self.request.session.modified = True

        # else browser session will be as long as the session cookie time "SESSION_COOKIE_AGE" defined in settings.py
        return super(CustomLoginView, self).form_valid(form)


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = "We've emailed you instructions for setting your password, " \
                      "if an account exists with the email you entered. You should receive them shortly." \
                      " If you don't receive an email, " \
                      "please make sure you've entered the address you registered with, and check your spam folder."
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')

from .models import Profile

def profile(request):
    user = request.user
    # Ensure the user has a profile
    if not hasattr(user, 'profile'):
        Profile.objects.create(user=user)
    
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        profile_form = UpdateProfileForm(request.POST, request.FILES, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='users-profile')
    else:
        user_form = UpdateUserForm(instance=request.user)
        profile_form = UpdateProfileForm(instance=request.user.profile)

    return render(request, 'users/profile.html', {'user_form': user_form, 'profile_form': profile_form})

from . models import UserImageModel
from . import forms
from .forms import UserImageForm


def Deploy_8(request): 
    print("HI")
    if request.method == "POST":
        form = forms.UserImageForm(files=request.FILES)
        if form.is_valid():
            print('HIFORM')
            form.save()
        obj = form.instance

        result1 = UserImageModel.objects.latest('id')C:\Users\Lenovo\Music\DL10\ITPDL11(Done)\Deploy\Project\App\keras_model.h5
        models = keras.models.load_model('C:\Users\Lenovo\Music\DL10\ITPDL11(Done)\Deploy\Project\App\forms.py')
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open("D:/Users/Lenovo/Downloads/DL10/DL10/ITPDL11(Done)/Deploy/Project/" + str(result1)).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        classes = ["MildDemented","ModerateDemented","Non_Parkinson","NonDemented","Parkinson","VeryMildDemented"]
        prediction = models.predict(data)
        idd = np.argmax(prediction)
        a = (classes[idd])
        if a == "MildDemented":
             b ='This image Detected in Mild_Demented'
        elif a == "ModerateDemented":
            b ='This image Detected in Moderate_Demented'
        elif a == "Non_Parkinson":
            b ='This image Detected in Non_Parkinson'
        elif a == "NonDemented":
            b ='This image Detected in Non_Demented'
        elif a == "Parkinson":
            b ='This image Detected in Parkinson'
        elif a == "VeryMildDemented":
            b ='This image Detected in Very_Mild_Impairment'
        
        else:
            b = 'WRONG INPUT'

        data = UserImageModel.objects.latest('id')
        data.label = a
        data.save()

        # engine = pyttsx3.init()
        # rate = engine.getProperty('rate')
        # engine.setProperty('rate', rate - 10)  # Decrease rate by 50 (default rate is typically around 200)
        # engine.say(a)
        # engine.runAndWait()

        
        # text_to_speech(a, delay=7)

        
        return render(request, 'App/output.html',{'form':form,'obj':obj,'predict':a,'predict1':b})
    else:
        
        form = forms.UserImageForm()
    return render(request, 'App/model.html',{'form':form})




def Database(request):
    models = UserImageModel.objects.all()
    return render(request, 'App/Database.html', {'models': models})


def logout_view(request):  
    auth_logout(request)
    return redirect('/')





Model = joblib.load('App/model_1.pkl')

    
def model(request): 
    if request.method == "POST":
        int_features = [x for x in request.POST.values()]
        int_features = int_features[1:]
        print(int_features)
        final_features = [np.array(int_features, dtype=object)]
        print(final_features)
        prediction = Model.predict(final_features)
        print(prediction)
        output = prediction[0]
        print(f'output{output}')
        if output == 0:
            a="normal"
           
        elif output == 1:
            a="Parkinson"
            
        return render(request, 'app/predict_out.html', {"prediction_text":a,'predict':output})
       
    else:
        return render(request, 'app/predict.html')













from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .forms import AudioForm
from .models import AudioPrediction
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import os

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)  # Adjusted to keep the original sample rate
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def play_model(request):
    
    return render(request, 'AU/Deploy_8.html')







def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)  # Adjusted to keep the original sample rate
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def model1(request):
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = request.FILES['audio_file']
            fs = FileSystemStorage()
            file_name = fs.save(audio_file.name, audio_file)
            file_path = fs.url(file_name)
            full_path = os.path.join(settings.MEDIA_ROOT, file_name)

            features = extract_features(full_path)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=-1)

            # Load the model
            model = load_model('E:/2025 - 2026/DL/brain done/ITPDL11(Done)/Deploy/Project/App/audio_classification_model.h5',compile=False)  # Ensure this path is correct

            # Predict the class
            predicted_class = model.predict(features)
            predicted_label = np.argmax(predicted_class)

            # Map predicted label to corresponding class
            class_mapping = {
                0: 'No_parkision',
                1: 'Parkision',     
            }
            prediction = class_mapping.get(predicted_label, 'unlabel')

            # Save the result in the database
            audio_prediction = AudioPrediction(audio_file=audio_file, prediction=prediction)
            audio_prediction.save()

            return render(request, 'AU/output.html', {'prediction': prediction, 'audio_file': audio_prediction})
        else:
            return render(request, 'AU/model.html', {'form': form})
    return render(request, 'AU/model.html')




from django.shortcuts import render
from django.http import JsonResponse
# import random
# import json
import numpy as np
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
#from .models import Response, models
from Chatbot.processor import chatbot_response
# Remove the comments to download additional nltk packages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

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
 

               
def bott(request):
    return render(request, 'chatbot/bott.html')



def model_db(request):
    
    models = AudioPrediction.objects.all()
    return render(request, 'AU/model_db.html', {'models':models})