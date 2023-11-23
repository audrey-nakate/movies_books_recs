from django.db import models





class Directors(models.Model):
    name = models.CharField(max_length=35)
    def __str__(self) :
        return self.name
    

class Authors(models.Model):
    name = models.CharField(max_length=35)
    def __str__(self) :
        return self.name


class Movies(models.Model):
    title = models.CharField('Title of the movie',max_length=100)
    director = models.ForeignKey(Directors,blank=True,null = True,on_delete=models.CASCADE)
    # director = models.CharField('Director of the movie',max_length=100)

    def __str__(self) :
        return self.title



class Books(models.Model):
    name = models.CharField('Name of the book',max_length=200)
    author = models.ForeignKey(Authors,blank=True,null = True,on_delete=models.CASCADE)
    # author = models.CharField('Name of the Author',max_length=150)
    publisher = models.CharField('Name of the publisher',max_length=150)
    description = models.CharField('A brief description of the book ',max_length=150)
    ISBN = models.IntegerField()

    def __str__(self) :
        return self.name



        

