from flask import Flask, render_template, request
import Combined_prediction2

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/display', methods=['POST'])
def display():
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    df = Combined_prediction2.combine(start_date,end_date)  # Assuming this function returns a DataFrame
    return render_template('display.html', start_date=start_date, end_date=end_date, df=df)

if __name__ == '__main__':
    app.run(debug=True)
