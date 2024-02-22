# Insurance Cross Sell ML Project - Lemonaade Insurance

This repo is for a group project done for Northwestern University's Master's of Data Science program 498 course. The goal of the project was to conduct data analysis and build a ML model with the end result of serving the model via an API and a client side UI. The hypothetical business case constructed was for an insurance providing company, "Lemonaade", to assess which factors contribute to customer costs (regression) and a (classification) to assess which customers/policy holders would be most likely to add insurance policies through cross selling marketing efforts. The source of data was a combination of the kaggle sets in the links below. The file 'combinedHealthPred.csv' within the 'data' directory in this repo contains the sample. 

https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction/data
https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction/data 

The project currently is a MVP, not production quality. Further work would be required in error handling, UI refinements, CI/CD, observability, etc. for more real world use case. 

## Dashboard
An interactive Tableau dashboard was created. Dashboard is [here](https://public.tableau.com/app/profile/ash.vaid/viz/498-capstone-2024-q1/Dashboard1#2).

# Running locally

Make sure you have "docker" and "docker compose" installed, which can be bundled together with [docker desktop](https://docs.docker.com/desktop/)



```

## from project root

docker compose up -d --build

## to spin down

docker compose down
```
Visit http://localhost:8501/ for streamlit built client. 
Backend API built with FastAPI, Swagger spec available at http://localhost:8000/docs

## Posting data

The "test.csv" file within the "data" directory can serve as a sample for uploading test data. The output JSON file response will contain a list of 1 (likely) and -1 (not likely) for those policy holders to target in terms of cross selling. 


