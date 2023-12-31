from django.conf import settings
from django.contrib.auth import views as auth_views
from django.urls import path
from django.conf.urls.static import static
from . import views
from .forms import LoginForm
from .views import index

app_name = 'core'

urlpatterns = [
    # path('', views.index, name='index'),
    path('', views.database, name='database'),
    path('signup/', views.signup, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='core/login.html', authentication_form=LoginForm), name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page=settings.LOGOUT_REDIRECT_URL), name='logout'),
    path('addbook/',views.addBooks,name='addbook'),
    path('book/<book_id>',views.book,name ='book')
    ]

urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)
