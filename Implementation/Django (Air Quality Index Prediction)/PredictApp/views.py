from django.shortcuts import render, redirect
from .models import User

from django.http import HttpResponse
from django.template.loader import render_to_string
from io import BytesIO
from xhtml2pdf import pisa
from PredictApp.utils.AirQualityPredictor import AirQualityPredictor

from django.templatetags.static import static

from django.conf import settings

def index(request):
    context = {
        'predictions': None,
        'aqi_images': {
            'Good': static('PredictApp/images/good.png'),
            'Satisfactory': static('PredictApp/images/satisfactory.png'),
            'Moderate': static('PredictApp/images/moderate.png'),
            'Poor': static('PredictApp/images/poor.png'),
            'VeryPoor': static('PredictApp/images/verypoor.png'),
            'Severe': static('PredictApp/images/severe.png'),
        }
    }
    if request.method == 'POST':
        fromdate = request.POST.get('fromDate')
        toDate = request.POST.get('toDate')
        aq_predictor = AirQualityPredictor(
            ann_model_path=settings.ANN_MODEL_PATH,
            lstm_model_path=settings.LSTM_MODEL_PATH,
            scaler_ann_path=settings.SCALER_ANN_PATH,
            scaler_lstm_path=settings.SCALER_LSTM_PATH,
            data_path=settings.DATA_PATH
        )
        result_df = aq_predictor.run_predictions(fromdate, toDate)
        result_df = result_df.rename(columns={"PM2.5": "PM2point5"})
        result_df = result_df.rename(columns={"AQI Prediction": "AQI"})
        
         # Convert DataFrame to a list of dictionaries for easier rendering in a Django template
        predictions_list = result_df.to_dict(orient='records')
         
        # Save the predictions to a CSV file
        output_path = settings.PREDICTION_OUTPUT_PATH
        aq_predictor.save_predictions(result_df, output_path)
        
        # Render the results using a template or return a JSON response
        # Update context with predictions
        context['predictions'] = predictions_list
        
        return render(request, 'index.html', context)

    else:
        return render(request, 'index.html')
    
    

def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate user against database
        try:
            user = User.objects.get(username=username, password=password)
            # If user exists and credentials match, redirect to next page
            return redirect('dashboard')
        except User.DoesNotExist:
            # If user does not exist or credentials do not match, show error message
            return render(request, 'signin.html', {'error': 'Invalid username or password'})

    else:
        return render(request, 'signin.html')

def dashboard(request):
    # Implement logic for the dashboard view
    return render(request, 'dashboard.html')

def logout(request):
    # Implement logic for the dashboard view
    return render(request, 'index.html')

def generate_pdf(request):
    # Retrieve data from the HTML form
    from_date = request.GET.get('fromDate')
    to_date = request.GET.get('toDate')
    html_content = render_to_string('index.html', {'from_date': from_date, 'to_date': to_date})

    # Create a PDF document
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="report.pdf"'

    # Parse HTML content and convert it into PDF
    pdf = BytesIO()
    pisa.CreatePDF(html_content, dest=pdf)

    # Write PDF to response
    pdf.seek(0)
    response.write(pdf.getvalue())
    return response