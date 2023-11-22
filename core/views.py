from django.shortcuts import redirect, render

from .forms import SignUpForm


def index(request):

    return render(request, 'core/index.html')

def signup(request):
    if request.method =='POST':   
        form = SignUpForm(request.POST)

        if form.is_valid():
             form.save()
             return redirect('login/')
    else:
      form = SignUpForm()       

      return render(request, 'core/signup.html', {
            'form' : form,
      })
    
