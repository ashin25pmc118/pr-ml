from flask import Flask, render_template, request, session
import csv
import io
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import pickle

app = Flask(__name__)
app.secret_key = 'plr-secret-key-2024'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        # STEP 1: Handle File Upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error_msg='No selected file.', step=1)

            try:
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
                file.save(filepath)

                with open(filepath, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    headers = next(csv_reader, None)
                    data_preview = [row for i, row in enumerate(csv_reader) if i < 10]

                return render_template('index.html', headers=headers, data_preview=data_preview, step=2)
            except Exception as e:
                return render_template('index.html', error_msg=f"Upload error: {e}", step=1)

        # STEP 3: Predict a value entered by the user (check BEFORE x_var/y_var branch)
        elif 'predict_x' in request.form:
            try:
                predict_x = float(request.form['predict_x'])
                x_var = request.form.get('x_var', 'X')
                y_var = request.form.get('y_var', 'Y')
                degree = int(request.form.get('degree', 2))
                accuracy = request.form.get('accuracy', '')
                plot_url = request.form.get('plot_url', '')

                model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl')
                poly_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'poly.pkl')

                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                with open(poly_path, 'rb') as f:
                    poly = pickle.load(f)

                X_input = np.array([[predict_x]])
                X_poly_input = poly.transform(X_input)
                predicted_y = round(float(model.predict(X_poly_input)[0]), 4)

                return render_template('index.html',
                                       plot_url=plot_url,
                                       accuracy=accuracy,
                                       degree=degree,
                                       x_var=x_var,
                                       y_var=y_var,
                                       predict_x=predict_x,
                                       predicted_y=predicted_y,
                                       step=3)
            except Exception as e:
                return render_template('index.html', error_msg=f"Prediction error: {e}", step=1)

        # STEP 2: Run Polynomial Regression
        elif 'x_var' in request.form and 'y_var' in request.form:
            x_var = request.form['x_var']
            y_var = request.form['y_var']
            degree = int(request.form.get('degree', 2))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')

            try:
                x_data, y_data = [], []
                with open(filepath, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    headers = list(next(csv_reader))
                    x_idx, y_idx = headers.index(x_var), headers.index(y_var)

                    for row in csv_reader:
                        try:
                            x_data.append(float(row[x_idx].strip()))
                            y_data.append(float(row[y_idx].strip()))
                        except (ValueError, IndexError):
                            continue

                X = np.array(x_data).reshape(-1, 1)
                y = np.array(y_data)

                # Polynomial Transformation
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(X)

                model = LinearRegression()
                model.fit(X_poly, y)
                y_pred = model.predict(X_poly)
                accuracy = round(r2_score(y, y_pred) * 100, 2)

                # Save model, poly, and meta to disk for prediction use
                model_path = os.path.join(app.config['UPLOAD_FOLDER'], 'model.pkl')
                poly_path  = os.path.join(app.config['UPLOAD_FOLDER'], 'poly.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                with open(poly_path, 'wb') as f:
                    pickle.dump(poly, f)

                # Generate plot
                X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
                y_range_pred = model.predict(poly.transform(X_range))

                plt.figure(figsize=(8, 5))
                plt.scatter(X, y, color='royalblue', alpha=0.6, label='Actual Data')
                plt.plot(X_range, y_range_pred, color='crimson', linewidth=3,
                         label=f'Polynomial Fit (Deg {degree})')
                plt.xlabel(x_var)
                plt.ylabel(y_var)
                plt.title(f'Polynomial Regression: {y_var} based on {x_var}')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)

                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close()

                return render_template('index.html',
                                       plot_url=plot_url,
                                       accuracy=accuracy,
                                       degree=degree,
                                       x_var=x_var,
                                       y_var=y_var,
                                       step=3)
            except Exception as e:
                return render_template('index.html', error_msg=f"Analysis error: {e}", step=1)



    return render_template('index.html', step=1)

if __name__ == '__main__':
    app.run(debug=True, port=8003)