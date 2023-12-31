from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from django.forms import ModelForm
from .models import Books

#Creating a book form that we can use to add books to the database without needing the admin page

class BookForm(ModelForm):
    class Meta:
        model = Books
        fields = ('name','author','publisher','description','ISBN')
        labels = {
            'name':'',
            # 'author':'',
            'publisher':'',
            'description':'',
            'ISBN':'',
        }
        widgets = {
            'name':forms.TextInput(attrs={'class':'form-control','placeholder':'Book title'}),
            # 'author':forms.TextInput(attrs={'class':'form-control','placeholder':'Book author'}),
            'publisher':forms.TextInput(attrs={'class':'form-control','placeholder':'Published by'}),
            'description':forms.TextInput(attrs={'class':'form-control','placeholder':'Brief description'}),
            'ISBN':forms.TextInput(attrs={'class':'form-control','placeholder':'ISBN'}),
        }

class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')

        username = forms.CharField(widget=forms.TextInput(attrs={
        'placeholder' : 'Your username',
    })) 

    email = forms.CharField(widget=forms.EmailInput(attrs={
        'placeholder' : 'Your email address',
    })) 

    password1 = forms.CharField(widget=forms.PasswordInput(attrs={
        'placeholder' : 'Your password',
    }))   

    password2 = forms.CharField(widget=forms.PasswordInput(attrs={
        'placeholder' : 'Repeat password',
    })) 

class LoginForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={
        'placeholder' : 'Your username',
    })) 

    password = forms.CharField(widget=forms.PasswordInput(attrs={
        'placeholder' : 'Your password',
    })) 