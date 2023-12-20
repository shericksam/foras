
import os 
import nltk
import json
import string
import PyPDF2
import requests
import torch
import docx
from nltk.corpus import stopwords
from django.shortcuts import render
from django.http import JsonResponse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from django.http import HttpResponse
from requests.auth import HTTPBasicAuth
from nltk.stem import PorterStemmer, WordNetLemmatizer
from torch.utils.data import Dataset, DataLoader
from torch import nn, cuda
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from django.shortcuts import render, redirect
from .models import Message, User
from .forms import MessageForm, SignUpForm
from django.contrib.auth import login
from django.contrib.auth import authenticate, login
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from .forms import DocumentUploadForm
from openai import OpenAI

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
model_directory = "./ai-model/"

class MyDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        text = self.texts[idx] 
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

def create_data_loader(texts, tokenizer, batch_size):
    dataset = MyDataset(texts, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_model(request):
    print("Training new model...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)

    return JsonResponse({'message': 'Model initialized.'})

def train_model(file_name):
    model = None
    tokenizer = None
    print("Loading saved model...")
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_directory)
    tokenizer.pad_token = tokenizer.eos_token
    
    with open("./static/foras/jsons/" +file_name + ".json", "r") as file:
        data = json.load(file)
    
    dataset = MyDataset(data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()  # or your loss function
    device = 'cuda' if cuda.is_available() else 'cpu'  
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
        
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            if loss is not None:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            else:
                print('No loss value to perform backpropagation on.')
        # model.eval()
        # with torch.no_grad():   
        #     total_loss, total_correct = 0, 0
            
        #     for batch in valid_loader:
        #         input_ids = batch["input_ids"].to(device)
        #         attention_mask = batch["attention_mask"].to(device)
        #         labels = batch["labels"].to(device)  
                
        #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    model.save_pretrained(model_directory)
    tokenizer.save_pretrained(model_directory)

    return JsonResponse({'message': 'Training complete'}, status=200)

def doc_procesing(request):
    username = "erguerrero"
    api_token = "your-api"
    base_url = "https://confluence/wiki/rest/api"
    page_id = "yyyyy"
    page_url = f"{base_url}/content/{page_id}"
    response = requests.get(page_url, auth=HTTPBasicAuth(username, api_token))
    data = response.json()

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text += page.extract_text()
    pdf_file.close()
    return text.strip()

def extract_text_from_word(word_file):
    doc = docx.Document(word_file)
    text = "\n".join(para.text for para in doc.paragraphs)
    return text.strip()

def read_doc(request):
    text = extract_text_from_pdf("./docs/file.pdf")
    return JsonResponse({'text': preprocess_text_token(text)}, status=200)

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    nltk.download('punkt')
    word_tokens = word_tokenize(text)
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    filtered_text = [word for word in word_tokens if word not in stop_words]
    processed_text = []
    for word in filtered_text:
        if word not in stop_words:
            stemmed_word = stemmer.stem(word)
            lemmatized_word = lemmatizer.lemmatize(stemmed_word)
            processed_text.append(lemmatized_word)
    return processed_text

def preprocess_text_token(text):
    sentences = sent_tokenize(text)
    processed_sentences = []
    
    for sentence in sentences:
        sentence = sentence.lower().translate(str.maketrans('', '', string.punctuation))
        word_tokens = word_tokenize(sentence)
        processed_words = [
            lemmatizer.lemmatize(stemmer.stem(word))
            for word in word_tokens
            if word not in stop_words
        ]
        processed_sentence = ' '.join(processed_words)
        processed_sentences.append(processed_sentence)
        
    return processed_sentences
    
def stem_and_lemmatize(word):
    stemmed_word = stemmer.stem(word)
    lemmatized_word = lemmatizer.lemmatize(word)
    return stemmed_word, lemmatized_word

def save_file(text, file_name):
    file_name = './static/foras/jsons/'+file_name+'.json'
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False, indent=4)
    return file_name

def processing_doc(request):
    return train_model(request.GET["file"])

@login_required(login_url='login/')
def chat_view(request):
    if request.method == "POST" and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        input_text = request.POST.get('message')
        print("input_text----->", input_text)
        response_text = generate_response(input_text, request)
        Message.objects.create(user=request.user, text=input_text, is_user=True)
        Message.objects.create(user=request.user, text=response_text, is_user=False)
        return JsonResponse({'message': response_text})
    
    if request.method == 'POST':
        form = MessageForm(request.POST)
        if form.is_valid():
            message = form.save(commit=False)
            message.user = request.user
            message.save()
            return redirect('chat')  
    else:
        form = MessageForm()
    
    messages = Message.objects.all().order_by('created_at')
    context = {
        'messages': messages,
        'form': form,
    }
    return render(request, 'chat.html', context)

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            user.first_name = form.cleaned_data.get('first_name')
            user.last_name = form.cleaned_data.get('last_name')
            user.email = form.cleaned_data.get('email')
            user.save()
            login(request, user)
            return redirect('chat')
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})

def generate_response(input_text, request):
    client = OpenAI()
    N = 5
    conversation_history = Message.objects.filter(user=request.user).order_by('-created_at')[:N]
    conversation_history = list(reversed(conversation_history))
    formatted_history = [
        {"role": "user" if message.is_user else "system", "content": message.text }
        for message in conversation_history
    ] 
    with open("./concepts.json", "r") as file:
        data = json.load(file)
        formatted_history + data

    formatted_history = formatted_history + [{"role": "user", "content": input_text}] 
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=formatted_history
    )
    return completion.choices[0].message.content

def generate_response_old(input_text):
    model_directory = "./ai-model/"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_directory)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )


    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    # output_sequences = model.generate(
        # input_ids=input_ids,
        # attention_mask=attention_mask,
    #     max_length=50,
    #     temperature=0.9,
    #     top_k=50,
    #     top_p=0.95,
    #     no_repeat_ngram_size=2,
    #     repetition_penalty=1.2,
    #     do_sample=True,
    #     num_return_sequences=1
    # )
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=50,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=2.0,
        do_sample=True,
        num_return_sequences=1
    )
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return generated_text

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('chat')
            else:
                form.add_error(None, 'user or password wrong.')
    else:
        form = AuthenticationForm()

    return render(request, 'login.html', {'form': form})

def upload_document(request):
    if request.method == 'POST':
        form = DocumentUploadForm(request.POST, request.FILES)
        if form.is_valid():
            document = request.FILES['document']
            if document.name.endswith('.pdf'):
                text = extract_text_from_pdf(document)
            elif document.name.endswith('.docx'):
                text = extract_text_from_word(document)
            else:
                return HttpResponse('Unsupported file type.')
            
            text_processed = preprocess_text_token(text)
            file_name = save_file(text_processed, document.name)
            return HttpResponse(text_processed)
    else:
        form = DocumentUploadForm()
    return render(request, 'upload_document.html', {'form': form})
# --------------------------------------------------------------------
import pandas as pd
def init_open():
    client = OpenAI()
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
    ]
    )

    print(completion.choices[0].message)

def learn():
    client = OpenAI()
    with open("file.jsonl", "rb") as file:
        print("file-->", file)
        response = client.fine_tuning.jobs.create(
            training_file=file, 
            model="gpt-3.5-turbo", 
        )
        print(response)

def upload_jsonl(request):
    client = OpenAI()
    file = client.files.create(
        file=open("./file.jsonl", "rb"),
        purpose="fine-tune"
    )
    client.fine_tuning.jobs.create(
        training_file=file.id,
        model="gpt-3.5-turbo"
    )

    return JsonResponse({'text': file.id}, status=200)
from collections import defaultdict
def errors():
    data_path = "file.jsonl"

    # Load the dataset
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f]

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)


    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
            
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
            
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
                
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")