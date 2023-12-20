from django import forms
from .models import Message
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class MessageForm(forms.ModelForm):
    class Meta:
        model = Message
        fields = ['text']  

class SignUpForm(UserCreationForm):
    first_name = forms.CharField(max_length=30, required=False, help_text='Optional.')
    last_name = forms.CharField(max_length=150, required=False, help_text='Optional.')
    email = forms.EmailField(max_length=255, help_text='Required. Enter a valid email address.')

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')
        
class DocumentUploadForm(forms.Form):
    document = forms.FileField()