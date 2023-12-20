from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("init/", views.create_model, name="init"),
    path('signup/', views.signup, name='signup'),
    path('login/', views.login_view, name='login'),
    path('upload-document/', views.upload_document, name='upload_document'),
    path('chat/', views.chat_view, name='chat'),
    # path('train/', views.train_model, name='train_model'),
    path('upload-jsonl/', views.upload_jsonl, name='upload_jsonl'),
    path('read-doc/', views.read_doc, name='read_doc'),
    path('processing-doc/', views.processing_doc, name='processing_doc'),
]