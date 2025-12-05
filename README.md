# AI Learning Lab 🚀

An interactive web application designed to help students learn about Artificial Intelligence through hands-on experimentation. Upload data, select AI models, train them, and see beautiful visualizations of the results!

## Features

✨ **User-Friendly Interface**: Beautiful, intuitive design with step-by-step guidance
📊 **Multiple Data Types**: Support for CSV (tabular data) and images
🧠 **Various AI Models**:
  - Linear Regression
  - Decision Tree (Classification & Regression)
  - Neural Networks
  - CNN Image Classifier

🎨 **Visualizations**:
  - Model performance metrics
  - Confusion matrices
  - Feature importance
  - Hidden layer activations (for images)

## Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd "c:\Users\Omega\Desktop\ai"
   ```

2. **The virtual environment is already created!** Just activate it:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

3. **All packages are already installed!** Including:
   - Flask
   - TensorFlow
   - scikit-learn
   - pandas
   - numpy
   - Pillow
   - matplotlib
   - seaborn

## Running the Application

1. **Activate the virtual environment** (if not already activated):
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. **Run the Flask application**:
   ```powershell
   python app.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## How to Use

### Step 1: Upload Your Data
- Click the upload area or drag and drop a file
- Supported formats: CSV files, PNG, JPG, JPEG images
- See a preview of your uploaded data

### Step 2: Choose Your AI Model
- Browse available AI models with friendly descriptions
- Select the one that fits your task
- For tabular data, select which column to predict and which features to use

### Step 3: Train Your Model
- Click "Start Training" and watch the AI learn!
- The model will process your data and learn patterns

### Step 4: View Results
- See beautiful visualizations of model performance
- Check accuracy, error metrics, and other statistics
- For images: Explore how the neural network "sees" your image through hidden layers!

## Example Datasets

You can try the app with sample datasets:

### For Tabular Data
Create a simple CSV file with data like:
- House prices (size, bedrooms, price)
- Student grades (study hours, quiz scores, final grade)
- Weather data (temperature, humidity, rainfall)

### For Images
Use any PNG or JPG image:
- Photos of objects
- Handwritten digits
- Nature scenes

## Project Structure

```
ai/
├── app.py                  # Flask backend application
├── templates/
│   └── index.html         # Main web interface
├── static/
│   ├── css/
│   │   └── style.css      # Styling and animations
│   └── js/
│       └── main.js        # Frontend JavaScript logic
├── uploads/               # Uploaded files (auto-created)
├── venv/                  # Virtual environment
└── README.md              # This file
```

## Technologies Used

- **Backend**: Flask (Python web framework)
- **Machine Learning**: 
  - scikit-learn (traditional ML models)
  - TensorFlow/Keras (deep learning)
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5

## Educational Features

This app is designed to make AI education accessible and fun:

1. **Visual Learning**: Every step shows what's happening
2. **Immediate Feedback**: See results right away
3. **No Coding Required**: Students can experiment without programming knowledge
4. **Hidden Layer Visualization**: Understand how neural networks process images
5. **Multiple Algorithms**: Compare different AI approaches

## Tips for Students

- Start with simple datasets to understand the basics
- Try different models on the same data to compare results
- For images, use the layer visualization to see what the AI "sees"
- Experiment with different features to see how they affect predictions

## Troubleshooting

**Port already in use?**
Edit `app.py` and change the port number:
```python
app.run(debug=True, port=5001)  # Change 5000 to 5001 or another port
```

**File too large?**
Maximum file size is 16MB. For larger datasets, try using a sample of your data.

**Model taking too long?**
Neural networks can take time to train. Start with smaller datasets or fewer epochs.

## Future Enhancements

- Real-time training progress
- Model comparison features
- More visualization options
- Support for more file formats
- Pre-trained model examples
- Interactive tutorials

## License

This project is created for educational purposes. Feel free to use and modify it for learning!

---

Made with ❤️ for AI education
