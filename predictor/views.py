import os
from django.shortcuts import render
import pickle
import numpy as np
from django.conf import settings
from django.template.loader import get_template
from django.http import HttpResponse


def predict_price(request):
    if request.method == 'POST':
        name = request.POST['name']
        fueltype = request.POST['fueltype']
        doornumber = request.POST['doornumber']
        drivewheel = request.POST['drivewheel']
        horsepower = float(request.POST['horsepower'])
        enginesize = float(request.POST['engine_size'])

        current_dir = os.getcwd()
        file_path = os.path.join(current_dir, 'car_price_model.pkl')
        with open(file_path, 'rb') as f:
            model = pickle.load(f)

        features = [[enginesize, horsepower]]

        predicted_price = model.predict(features)[0]
        predicted_price = np.round(predicted_price, 2)

        return render(request, 'result.html', {'predicted_price': predicted_price})

    return render(request, 'predictor/index.html')
