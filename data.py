from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Dummy model initialization and fitting for demonstration purposes
model = LinearRegression()
# For the purpose of this example, some arbitrary training data is used
X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y_train = np.array([[10, 5], [20, 10], [30, 15]])
model.fit(X_train, y_train)

@app.route('/')
def home():
    # Render the home page
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            close_last = float(request.form['close_last'])
            volume = float(request.form['volume'])
            open_price = float(request.form['open_price'])

            input_data = np.array([[close_last, volume, open_price]])
            predicted_prices = model.predict(input_data)

            # Assuming the model predicts two values: high and low prices
            predicted_high_price = predicted_prices[0][0]
            # Note: You don't have a 'low_price' field in the result template. If you want to show it, add it there too.

            # Pass the prediction to the 'result.html' template using string formatting directly
            return render_template('result.html', high_price=f'{predicted_high_price:.2f}')
       
        except Exception as e:
            # Consider adding an error.html or modify 'result.html' to handle errors
            return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
