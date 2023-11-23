from django.shortcuts import redirect, render
from .models import Books
from .models import Movies
from .forms import SignUpForm,BookForm
from django.http import HttpResponseRedirect


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
    

def book(request,book_id):
    individualbook = Books.objects.get(pk=book_id)
    return render(request,'core/book.html',
                  {'individualbook':individualbook})    

def database(request):
    books_list = Books.objects.all()
    return render(request, 'core/books.html',
                  {'books_list':books_list})

def addBooks(request):
    submitted = False
    if request.method == "POST":
        form = BookForm(request.POST)
        if form.is_valid():
            form.save()
            return HttpResponseRedirect('/addbook?submitted=True')
    else:
        form = BookForm
        if 'submitted' in request.GET:
            submitted = True
    return render(request , 'core/addbooks.html',{'form':form,'submitted':submitted},)
