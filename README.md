

```markdown
# Healthcare Insurance Claim Fraud Detection System

## [cite_start]1. Project Overview [cite: 156]
[cite_start]This project is a production-ready Machine Learning API service designed to predict whether a healthcare insurance claim is legitimate or potentially fraudulent[cite: 4, 62]. [cite_start]It utilizes a **Random Forest Classifier** trained on the Healthcare Provider Fraud Detection Analysis dataset[cite: 29, 47].

## [cite_start]2. Technical Requirements [cite: 91]
- [cite_start]**API Framework**: FastAPI [cite: 94]
- [cite_start]**Model**: Random Forest Classifier (scikit-learn) [cite: 47, 99]
- [cite_start]**Containerization**: Docker [cite: 65, 131]
- [cite_start]**CI/CD**: GitHub Actions [cite: 66, 105]
- [cite_start]**Quality Standards**: Ruff (PEP8), Bandit (Security), Pytest (Testing) [cite: 106, 110, 114]

## [cite_start]3. How to Run Locally [cite: 157]
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

```

2. Train the model:
```bash
python model/train.py

```


3. Start the API:
```bash
uvicorn app.main:app --reload

```



4. Docker Instructions 

To build the image locally:

```bash
docker build -t healthcare-fraud-app .

```

To run the container (accessible at port 8000):

```bash
docker run -p 8000:8000 healthcare-fraud-app

```

5. API Testing (curl commands) 

**Health Check:**

```bash
curl -X GET http://localhost:8000/health

```

**Prediction Endpoint:**

```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"InscClaimAmtReimbursed": 500.0, "DeductibleAmtPaid": 50.0, "IsInpatient": 1}'

```

6. Docker Hub URL 

The automated production image can be found here:
`https://hub.docker.com/r/<YOUR_DOCKERHUB_USERNAME>/healthcare-fraud`

7. CI/CD Explanation 

The project uses a GitHub Actions pipeline that automatically:

1. 
**Lints** the code using **Ruff** for PEP8 compliance.


2. 
**Scans** for security vulnerabilities using **Bandit**.


3. 
**Tests** the application with **Pytest**, ensuring a minimum of **80% coverage**.


4. 
**Builds and Pushes** the Docker image to Docker Hub upon successful validation.



```

---


```
