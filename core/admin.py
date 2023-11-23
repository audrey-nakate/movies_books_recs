from django.contrib import admin
from .models import Directors
from .models import Authors
from .models import Books
from .models import Movies

admin.site.register(Directors)
admin.site.register(Authors)
admin.site.register(Books)
admin.site.register(Movies)
