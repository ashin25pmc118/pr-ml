from flask import Flask, render_template, request
import csv
import io
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)

# Create a folder to temporarily store the uploaded CSV
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        # ==========================================
        # STEP 1: Handle File Upload & Read Headers + Top 10 Rows
        # ==========================================
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error_msg='No selected file.', step=1)
            
            try:
                # Save the file locally so we can access it in Step 2
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
                file.save(filepath)
                
                # Open the file to read the headers AND the first 10 rows
                with open(filepath, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    headers = next(csv_reader, None)
                    
                    if not headers:
                        raise ValueError("The uploaded CSV file is empty.")
                    
                    # Read the first 10 rows for the preview table
                    data_preview = []
                    for i, row in enumerate(csv_reader):
                        if i < 10:
                            data_preview.append(row)
                        else:
                            break
                    
                # Render the page again, showing dropdowns AND the 10-row preview (Step 2)
                return render_template('index.html', headers=headers, data_preview=data_preview, step=2)
                
            except Exception as e:
                return render_template('index.html', error_msg=f"Error reading file: {e}", step=1)

        # ==========================================
        # STEP 2: Process Selected Variables & Plot
        # ==========================================
        elif 'x_var' in request.form and 'y_var' in request.form:
            x_var = request.form['x_var']
            y_var = request.form['y_var']
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    headers = next(csv_reader)
                    
                    # Find which column index matches the user's choice
                    x_idx = headers.index(x_var)
                    y_idx = headers.index(y_var)
                    
                    x_data = []
                    y_data = []
                    
                    for row in csv_reader:
                        # Ensure the row has enough columns
                        if len(row) > max(x_idx, y_idx):
                            try:
                                x_data.append(float(row[x_idx].strip()))
                                y_data.append(float(row[y_idx].strip()))
                            except ValueError:
                                pass # Skip text/empty rows
                
                # Convert to numpy arrays
                X = np.array(x_data).reshape(-1, 1)
                y = np.array(y_data)
                
                if len(X) == 0:
                    raise ValueError(f"Could not find valid numbers in '{x_var}' or '{y_var}'. Try picking different columns.")

                # Machine Learning Model
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                accuracy = round(r2_score(y, y_pred) * 100, 2)
                
                # Generate Plot
                plt.figure(figsize=(8, 5))
                plt.scatter(X, y, color='blue', label='Actual Data', alpha=0.6)
                plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
                plt.xlabel(x_var)
                plt.ylabel(y_var)
                plt.title(f'{y_var} based on {x_var}')
                plt.legend()
                plt.grid(True)
                
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close()
                
                # Show the final results (Step 3)
                return render_template('index.html', plot_url=plot_url, accuracy=accuracy, step=3)
                
            except Exception as e:
                return render_template('index.html', error_msg=f"Analysis error: {e}", step=1)

    # Default to Step 1 on first load
    return render_template('index.html', step=1)

if __name__ == '__main__':
    app.run(debug=True)