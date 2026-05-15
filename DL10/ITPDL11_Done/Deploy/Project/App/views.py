from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout as auth_logout
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.storage import FileSystemStorage
from django.conf import settings

import numpy as np
import joblib
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import librosa

from .forms import RegisterForm, LoginForm, UpdateUserForm, UpdateProfileForm, UserImageForm, AudioForm
from .models import UserImageModel, Profile, AudioPrediction
from . import forms

from Chatbot.processor import chatbot_response


# ─── HOME ───────────────────────────────────────────────────────────────────

def home(request):
    return render(request, 'users/home.html')


# ─── INDEX ──────────────────────────────────────────────────────────────────

@login_required(login_url='users-register')
def index(request):
    return render(request, 'app/index.html')


# ─── REGISTER ───────────────────────────────────────────────────────────────

class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect(to='/')
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


# ─── LOGIN ──────────────────────────────────────────────────────────────────

class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')
        if not remember_me:
            self.request.session.set_expiry(0)
            self.request.session.modified = True
        return super(CustomLoginView, self).form_valid(form)


# ─── PASSWORD RESET / CHANGE ─────────────────────────────────────────────────

class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = (
        "We've emailed you instructions for setting your password, "
        "if an account exists with the email you entered. You should receive them shortly."
        " If you don't receive an email, "
        "please make sure you've entered the address you registered with, and check your spam folder."
    )
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')


# ─── PROFILE ────────────────────────────────────────────────────────────────

def profile(request):
    user = request.user
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


# ─── BRAIN MRI DETECTION ─────────────────────────────────────────────────────

def is_valid_brain_scan(image_path):
    """
    Rejects obviously non-MRI images using pixel statistics.
    Brain MRI scans are:
      - Near-grayscale (very low colour saturation)
      - Predominantly dark background
      - At least 20% near-black pixels
    Random photos, screenshots, and documents fail these checks.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.asarray(img, dtype=np.float32)
 
    # Check 1: colour saturation — MRIs are essentially grayscale.
    # Per-pixel spread between max and min channel should be small.
    channel_spread = arr.max(axis=2) - arr.min(axis=2)
    mean_spread    = channel_spread.mean()
    if mean_spread > 30:
        return False, "The uploaded image appears to be a colour photo, not a brain MRI scan."
 
    # Check 2: overall brightness — MRIs are predominantly dark.
    luminance = arr.mean(axis=2)
    mean_lum  = luminance.mean()
    if mean_lum > 160:
        return False, "The image is too bright to be a brain MRI scan."
 
    # Check 3: dark-pixel ratio — MRIs have large black backgrounds.
    dark_ratio = (luminance < 30).sum() / luminance.size
    if dark_ratio < 0.20:
        return False, "The image does not have the dark background characteristic of brain MRI scans."
 
    return True, ""
 
 
def Deploy_8(request):
    if request.method == "POST":
 
        # ── Guard 1: no file attached ──
        if 'image' not in request.FILES or not request.FILES['image']:
            return render(request, 'app/model.html', {
                'form': forms.UserImageForm(),
                'error_message': 'No image file was submitted. Please select an MRI scan.',
            })
 
        form = forms.UserImageForm(files=request.FILES)
        if not form.is_valid():
            return render(request, 'app/model.html', {
                'form': form,
                'error_message': 'Invalid file. Please upload a valid image (JPG, PNG, BMP, TIFF).',
            })
 
        form.save()
        obj     = form.instance
        result1 = UserImageModel.objects.latest('id')
 
        # ── Guard 2: pixel-level MRI validation (runs BEFORE the model) ──
        valid, reason = is_valid_brain_scan(result1.image.path)
        if not valid:
            result1.delete()   # clean up the bad upload from DB and disk
            return render(request, 'app/model.html', {
                'form': forms.UserImageForm(),
                'error_message': f'Invalid scan: {reason} Please upload a proper brain MRI image.',
            })
 
        # ── Load model & predict ──
        model_path = os.path.join(settings.BASE_DIR, 'App', 'keras_model.h5')
        model      = keras.models.load_model(model_path)
 
        data  = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = Image.open(result1.image.path).convert("RGB")
        image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)
        data[0] = (np.asarray(image).astype(np.float32) / 127.0) - 1
 
        classes    = ["MildDemented", "ModerateDemented", "Non_Parkinson",
                      "NonDemented", "Parkinson", "VeryMildDemented"]
        prediction = model.predict(data)
        confidence = float(np.max(prediction))
 
        # ── Guard 3: low model confidence (secondary safety net) ──
        if confidence < 0.60:
            result1.delete()
            return render(request, 'app/model.html', {
                'form': forms.UserImageForm(),
                'error_message': 'Low confidence: The model could not confidently classify this scan. Please ensure it is a clear, axial brain MRI image.',
            })
 
        a = classes[np.argmax(prediction)]
 
        labels = {
            "MildDemented":    "This image Detected in Mild_Demented",
            "ModerateDemented":"This image Detected in Moderate_Demented",
            "Non_Parkinson":   "This image Detected in Non_Parkinson",
            "NonDemented":     "This image Detected in Non_Demented",
            "Parkinson":       "This image Detected in Parkinson",
            "VeryMildDemented":"This image Detected in Very_Mild_Impairment",
        }
        b = labels.get(a, "WRONG INPUT")
 
        result1.label = a
        result1.save()
 
        return render(request, 'app/output.html', {
            'form':     form,
            'obj':      obj,
            'predict':  a,
            'predict1': b,
        })
 
    form = forms.UserImageForm()
    return render(request, 'app/model.html', {'form': form})
 

# ─── DATABASE ────────────────────────────────────────────────────────────────

def Database(request):
    models = UserImageModel.objects.all()
    return render(request, 'app/Database.html', {'models': models})


# ─── LOGOUT ──────────────────────────────────────────────────────────────────

def logout_view(request):
    auth_logout(request)
    return redirect('/')


# ─── PARKINSON FORM MODEL ────────────────────────────────────────────────────

Model = joblib.load('App/model_1.pkl')

def model(request):
    if request.method == "POST":
        int_features = [x for x in request.POST.values()]
        int_features = int_features[1:]
        final_features = [np.array(int_features, dtype=object)]
        prediction = Model.predict(final_features)
        output = prediction[0]
        a = "normal" if output == 0 else "Parkinson"
        return render(request, 'app/predict_out.html', {"prediction_text": a, 'predict': output})
    else:
        return render(request, 'app/predict.html')


# ─── AUDIO MODEL ─────────────────────────────────────────────────────────────

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


def play_model(request):
    return render(request, 'AU/Deploy_8.html')


def model1(request):
    if request.method == 'POST':
        form = AudioForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = request.FILES['audio_file']
            fs = FileSystemStorage()
            file_name = fs.save(audio_file.name, audio_file)
            full_path = os.path.join(settings.MEDIA_ROOT, file_name)

            features = extract_features(full_path)
            features = np.expand_dims(features, axis=0)
            features = np.expand_dims(features, axis=-1)

            audio_model_path = os.path.join(settings.BASE_DIR, 'App', 'audio_classification_model.h5')
            audio_model = load_model(audio_model_path, compile=False)

            predicted_class = audio_model.predict(features)
            predicted_label = np.argmax(predicted_class)

            class_mapping = {0: 'No_parkision', 1: 'Parkision'}
            prediction = class_mapping.get(predicted_label, 'unlabel')

            audio_prediction = AudioPrediction(audio_file=audio_file, prediction=prediction)
            audio_prediction.save()

            return render(request, 'AU/output.html', {'prediction': prediction, 'audio_file': audio_prediction})
        else:
            return render(request, 'AU/model.html', {'form': form})
    return render(request, 'AU/model.html')


# ─── CHATBOT ─────────────────────────────────────────────────────────────────

@require_POST
@csrf_exempt
def chatbot_response_view(request):
    the_question = request.POST.get('question', '')
    response = chatbot_response(the_question)
    return JsonResponse({"response": response})


def bott(request):
    return render(request, 'chatbot/bott.html')


# ─── AUDIO DATABASE ──────────────────────────────────────────────────────────

def model_db(request):
    models = AudioPrediction.objects.all()
    return render(request, 'AU/model_db.html', {'models': models})
