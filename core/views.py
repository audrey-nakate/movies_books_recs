from django.shortcuts import redirect, render
from django.urls import reverse

from .forms import SignUpForm


def index(request):
    return render(request, 'core/index.html')

def signup(request):
    if request.method =='POST':   
        form = SignUpForm(request.POST)

        if form.is_valid():
             form.save()
             return redirect('core:login')
        else:
            # If the form is not valid, render the signup page with the form and error messages
            return render(request, 'core/signup.html', {'form': form})
    else:
      form = SignUpForm()       

      return render(request, 'core/signup.html', {
            'form' : form,
      })
    
