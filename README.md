# LungIQ: Advanced Lung Disease Detection

**LungIQ** is an intelligent platform designed to analyze chest X-rays and detect lung conditions, including pneumonia, COVID-19, or healthy lungs. Leveraging state-of-the-art deep learning models, LungIQ delivers fast, accurate, and reliable diagnostic results, making it an invaluable tool for healthcare professionals and individuals alike.

---

## **Why Choose LungIQ?**
- **💡 Effortless Diagnosis:** Simply upload your chest X-rays for an instant and detailed evaluation.  
- **🎯 AI-Powered Precision:** Our deep learning models, built on CNNs, ensure accurate and trustworthy results.  
- **⚡ Time-Saving Insights:** Immediate diagnosis allows for timely medical attention and better outcomes.  
- **📖 Comprehensive Knowledge:** Learn about your condition with detailed information on symptoms and treatment options.

---

## **How It Works**

### **1. Upload the X-Ray Image**  
Submit a clear, high-resolution X-ray image of the lungs. A good-quality image improves diagnostic accuracy.

### **2. AI Analysis**  
The image is analyzed using a deep learning model specifically trained to classify lung conditions.

### **3. Receive Your Diagnosis**  
The platform provides a detailed report showing whether the lungs are healthy, or show signs of pneumonia or COVID-19.

---

## **Key Features**
- **🎯 High Accuracy:** Advanced CNNs ensure highly accurate predictions for every image.
- **⚡ Instant Results:** Obtain quick results to aid in timely medical decisions.
- **📖 Disease Insights:** Gain clarity on symptoms, treatments, and disease progression.

---

## **Technologies That Power LungIQ**

### **AI & Machine Learning Tools**
| Technology      | Role                                                                                  |
|------------------|---------------------------------------------------------------------------------------|
| Python           | Core programming language for building AI pipelines.                                |
| TensorFlow       | Framework used for creating and training CNN models.                                |
| NumPy            | Facilitates numerical operations and data preprocessing for model training.         |
| OpenCV           | Used for X-ray image processing and enhancement before feeding into the model.      |

### **Web Development Frameworks**
| Technology       | Role                                                                                 |
|-------------------|--------------------------------------------------------------------------------------|
| Flask            | Backend framework for building the application and managing APIs.                   |
| HTML5, CSS3      | Provides structure and styling for the frontend interface.                          |
| TailwindCSS      | Simplifies and accelerates the development of responsive and modern UIs.            |

---

## **Setup Instructions**

### **1. Prerequisites**
- Python 3 installed on your system.  
- MongoDB installed and running locally.

### **2. Clone the Project**
Run the following commands in your terminal:
```bash
git clone https://github.com/your-username/lungiq.git
cd lungiq

#### a) Install Python Packages
Install the necessary Python packages using pip:
```
pip install -r requirements.txt
```

Or manually:

```
pip install Flask pymongo opencv-python tensorflow werkzeug
```

#### b) Set Up MongoDB
Make sure MongoDB is running on your machine:

- Install MongoDB if you haven't already.
- Start MongoDB by running mongod in your terminal (or follow MongoDB's installation guide for your OS).

### **3. Run the Application**
Run the Flask app:
```
python app.py
```

### **4. Open the App in Your Browser**
Open your browser and go to:
```
http://127.0.0.1:5000/
```

You can now use the LungIQ web app!
