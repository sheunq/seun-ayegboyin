

export type Project = {
  id: number;
  slug: string;
  title: string;
  description: string;
  longDescription: string;
  images: string[];
  tags: string[];
  link: string;
  sourceCodeLink?: string;
  aiHint: string;
  sentimentAnalysisSection?: {
    title: string;
    content: string;
  };
  code?: string;
};

export type ProjectCategory = {
  id: number;
  slug: string;
  title: string;
  description: string;
  image: string;
  aiHint: string;
}

export type BlogPost = {
  id: number;
  slug: string;
  title: string;
  summary: string;
  content: string;
  author: string;
  date: string;
  image: string;
  aiHint: string;
};

const projectCategories: ProjectCategory[] = [
  {
    id: 1,
    slug: 'Data-Science',
    title: 'Data Science',
    description: 'Projects focused on machine learning, predictive modeling, and data analysis.',
    image: '/images/machinelearning.png',
    aiHint: 'data analytics',
  },
  {
    id: 2,
    slug: 'Visualization',
    title: 'Visualization',
    description: 'Projects showcasing data visualization techniques and dashboard creation.',
    image: '/images/Visualization.png',
    aiHint: 'financial chart',
  },
  {
    id: 3,
    slug: 'WebScraping',
    title: 'Web Scraping',
    description: 'Projects involving web scraping and data extraction from various sources.',
    image: '/images/Web Scraping.png',
    aiHint: 'source code',
  },
  {
    id: 4,
    slug: 'Database',
    title: 'Big Data/Database',
    description: 'Projects related to database management and big data technologies.',
    image: '/images/bigdata.png',
    aiHint: 'database servers',
  },
]

const projects: Project[] = [
  {
    id: 1,
    slug: 'TELCO-CHURN',
    title: 'Telco Churn',
    description: "Telco churn refers to the rate at which customers discontinue their services with a telecommunications company-whether it's mobile, internet, cable TV, or bundled services. It's a critical KPI in the telecom industry because high churn rates = revenue leakage + brand instability.",
    longDescription: "<h4>Project Overview</h4><p>Customers who left within the last month - the column is called Churn Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies Customer account information - how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges Demographic info about customers - gender, age range, and if they have partners and dependents Inspiration To explore this type of models and learn more about the subject.</p>",
    images: ['/images/Telco churn.png'],
    tags: ['Data-Science', 'Python', 'Scikit-learn',"Pandas","Numpy","Tensorflow","Keras","Pytorch"],
    link: '/project/customer-churn-prediction',
    sourceCodeLink: 'https://colab.research.google.com/drive/1WVol9PMr_TP3oyXUuAM2CIEku8CaF31q?usp=sharing#scrollTo=Su9JFe6BXFs9',
    aiHint: 'data analytics',
    sentimentAnalysisSection: {
// Suggested code may be subject to a license. Learn more: ~LicenseLog:1192901264.
      title: 'Project Overview',
      content: "Customers who left within the last month - the column is called Churn Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies Customer account information - how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges Demographic info about customers - gender, age range, and if they have partners and dependents Inspiration To explore this type of models and learn more about the subject."
    },
    code: `
import sys
assert sys.version_info>=(3,5)
import sklearn
assert sklearn.__version__ >= "0.20"

import os
import pandas as pd
import numpy as np


%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes',labelsize=14)
mpl.rc('xtick',labelsize=12)
mpl.rc('ytick',labelsize=12)


image_path=os.path.join('.','images','all_image')
os.makedirs(image_path,exist_ok=True)

def savefig(fig_id,tight_layout=True,resolution=300,fig_extension='png'):
  path=os.path.join(image_path,fig_id+'.'+fig_extension)
  print(f'saving figure {fig_id}')
  if tight_layout:
    plt.tight_layout()

  plt.savefig(path,format=fig_extension,dpi=resolution)

# Load the dataset
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.head()

# Preprocessing
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_train_full, test_size=0.33, random_state=11)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")
    `
  },
  {
    id: 2,
    slug: 'supply-chain-spend-analytics-dashboard-overview',
    title: 'supply chain spend analytics dashboard overview',
    description: 'This project involved building an interactive Spend Analytics Dashboard to help logistics, procurement and finance teams gain real-time visibility into organizational spending. The dashboard consolidated multi-year spend data across vendors, product categories, and logistics channels - providing a centralized decision-support system for cost control and strategic sourcing.',
    longDescription: `<section style="font-family: Arial, sans-serif; padding: 20px; max-width: 960px; margin: auto; line-height: 1.6;">
  <h2 style="color: #2c3e50;">📊 Supply Chain / Spend Analytics Dashboard Overview</h2>
  <p><strong>Focus Areas:</strong> Shipping Methods 🚚 | Product Categories 📦 | Supplier Performance 🧾<br>
     <strong>Period Covered:</strong> 2021–2023</p>

  <hr style="margin: 20px 0;">

  <h3 style="color: #34495e;">✈️ 1. Shipping Method Spend</h3>
  <ul>
    <li><strong>Air:</strong> $9.87M (highest spend)</li>
    <li><strong>Ground:</strong> $8.48M</li>
    <li><strong>Sea:</strong> $9.11M</li>
  </ul>
  <p><strong>📌 Insight:</strong><br>
    Air shipping leads in spend but only slightly edges out sea and ground. Assess trade-offs: speed vs. cost. Air may offer speed, but is the premium worth it?
  </p>

  <h3 style="color: #34495e;">📦 2. Spend by Product Subcategory</h3>
  <ul>
    <li><strong>Top Subcategory:</strong> “QM” – $2.1M 💰 (likely high-priority)</li>
    <li><strong>Others:</strong> Range from $0.1M–$1.5M</li>
  </ul>
  <p><strong>📌 Insight:</strong><br>
    “QM” commands the largest share. Prioritize procurement optimization, supplier terms, and accurate demand forecasting here.
  </p>

  <h3 style="color: #34495e;">🧾 3. Supplier Spend Analysis</h3>
  <ul>
    <li><strong>#1:</strong> C King-Rodriguez – $934.64K (9.47% of total) 🥇</li>
    <li><strong>Others:</strong> (e.g., Smith PLC, White and Sons): $400K–$600K (≈4–6% each)</li>
    <li><strong>Long Tail:</strong> Dozens of smaller suppliers contributing &lt;5% individually</li>
  </ul>
  <p><strong>📌 Insight:</strong><br>
    Consider supplier consolidation. Strengthening ties with high-spend partners may yield cost savings and improve operational synergy.
  </p>

  <h3 style="color: #34495e;">📦🔄 4. Total Stock Received by Supplier</h3>
  <ul>
    <li>Repetitive data labels suggest unclear correlation</li>
    <li>Spend vs. volume delivered needs clarity</li>
  </ul>
  <p><strong>📌 Action:</strong><br>
    Correlate inventory volumes with spend. Spot mismatches like high spend/low volume (or vice versa) to fix overstocking or supplier inefficiencies.
  </p>

  <hr style="margin: 30px 0;">

  <h3 style="color: #2c3e50;">📌 Key Recommendations</h3>
  <ul>
    <li><strong>🚀 Optimize Shipping:</strong> Assess if air freight's premium brings enough value. Shift toward ground/sea where feasible.</li>
    <li><strong>🤝 Supplier Rationalization:</strong> Focus on key suppliers like C King-Rodriguez for volume leverage, better pricing, and streamlined onboarding.</li>
    <li><strong>🎯 Product Subcategory Focus:</strong> Double down on “QM” — deploy forecasting tools and dynamic reorder strategies to stay ahead.</li>
    <li><strong>📉 Clarify Data Gaps:</strong> Refine metrics around stock received to ensure inventory aligns with actual spend and consumption.</li>
  </ul>
</section>
`,
    images: [
      '/images/supplychain2021air.png',
      '/images/supplychain2021ground.png',
      '/images/supplychain2021sea.png',
      '/images/supplychain2022sea.png',
    ],
    tags: ['Visualization', 'PowerBI'],
    link: '/project/sales-forecasting',
    aiHint: 'financial chart',
  },
  {
    id: 3,
    slug: 'scraping-airbnb',
    title: 'Scraping Airbnb',
    description: 'This project demonstrates my ability to combine web scraping techniques with machine learning to classify images. The goal was to build a pipeline that scrapes images of shoes from a website and uses them for training a machine learning model.',
    longDescription: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Scraping and Image Classification with TensorFlow</title>
</head>
<body>
    <h2>Web Scraping and Image Classification with TensorFlow</h2>
    <p>This project demonstrates my ability to combine web scraping techniques with machine learning to classify images. The goal was to build a pipeline that scrapes images of shoes from a website and uses them for training a machine learning model.</p>
    
    <h3>Key Features:</h3>
    <ul>
        <li><strong>Web Scraping:</strong> I utilized <code>BeautifulSoup</code> and <code>requests</code> to scrape images from an e-commerce site, focusing on categories like formal shoes, sneakers, and boots. The images were automatically downloaded and saved into organized directories.</li>
        <li><strong>Image Preprocessing:</strong> The images were stored in separate folders for training and testing. This organization sets the foundation for an image classification task.</li>
        <li><strong>Model Setup:</strong> While the project focuses on scraping, the next logical step (which can be explored further) is to use <code>TensorFlow</code> and <code>Keras</code> to create a Convolutional Neural Network (CNN) model to classify these images into categories such as sneakers, formal shoes, and boots.</li>
        <li><strong>Scalability:</strong> The design of the scraping function allows it to easily scale for other shoe categories or even other types of products. Simply by adjusting the URLs and folder names, new categories can be added to the pipeline.</li>
    </ul>
    
    <h3>Technologies Used:</h3>
    <ul>
        <li><strong>Python:</strong> The core language for web scraping and model building.</li>
        <li><strong>TensorFlow & Keras:</strong> For machine learning model development.</li>
        <li><strong>BeautifulSoup & Requests:</strong> For scraping data from the web.</li>
        <li><strong>PIL & Matplotlib:</strong> For handling images and visualizing results.</li>
    </ul>
    
    <h3>Why This Project?</h3>
    <p>This project not only helped me refine my web scraping skills but also gave me hands-on experience in integrating web data with machine learning models. It's an excellent example of how data from websites can be harvested and used in AI-driven applications, such as in e-commerce for product categorization.</p>
</body>
</html>
`,
    images: ['/images/airbnb.png'],
    tags: ['WebScraping', 'BeautifulSoup', 'Selenium', 'Scrapy'],
    link: '/project/sentiment-analysis-reviews',
    aiHint: 'cinema film',
    sourceCodeLink: 'https://github.com/sheunq/Scrapin-Airbnb/tree/main',
    sentimentAnalysisSection: {
      title: '💼 Business Impact',
      content: `This project has the potential to significantly enhance business operations, especially for e-commerce platforms. By automating the categorization and classification of product images, businesses can save time and resources that would otherwise be spent on manual labor. Here are some key business impacts:

      \n
Improved Product Categorization: Automated image classification ensures accurate categorization of products, which improves the shopping experience for customers. This makes it easier for businesses to organize their inventory and for customers to find what they are looking for quickly.
\n
Enhanced Customer Experience: By having accurate and consistent product categories, customers can easily navigate through various types of shoes (e.g., sneakers, formal shoes, boots), leading to improved satisfaction and higher conversion rates.`

    },
    code: `

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
from IPython.display import display
import pandas as pd
import re


# Load Data/webscraping

dairbnb=webdriver.Firefox()
airbnb.get('https://www.airbnb.com/')

sleep(10)

airbnb.refresh()

sleep(10)

try:
    
    #/html/body/div[10]/div/div/section/div
    airbnb.find_element(By.XPATH,"/html/body/div[10]/div/div/section/div/div/div[2]/div/div[1]/button/span").click()
except:
    airbnb.find_element(By.XPATH,"//*[text()='Rooms']").click() 
sleep(10)
airbnb.find_element(By.XPATH,'//*[text()="Show more"]').click()
sleep(10)

airbnb.execute_script("window.scrollTo(0, 0)")
sleep(10)

stopScrolling = 0
while True:
        stopScrolling += 1
        airbnb.execute_script("window.scrollBy(0,100)")
        sleep(0.5)
        if stopScrolling > 120:
            break
sleep(3)

airbnb.find_element(By.XPATH,'//*[text()="Show more"]').click()



all_room=airbnb.find_elements(By.XPATH,'//*[@id="site-content"]/div[2]/div[1]/div/div/div/div[1]/div')

rooms=[]
for i in range(1,len(all_room)):
    room=airbnb.find_element(By.XPATH,f'//*[@id="site-content"]/div[2]/div[1]/div/div/div/div[1]/div[{i}]/div/div[2]/div/div/div/div/div/div[2]')
    rooms.append(room.text)

#Data Scraping

data=pd.DataFrame(z)[[0,1,3,7,9,13]]
data[["Location","Owner","employment status","Date","Price","Rating"]]=data[[0,1,3,7,9,13]]
data.drop(data[[0,1,3,7,9,13]],axis=1)


` 
  },
  {
    id: 4,
    slug: 'Transaction-Customers-Analysis',
    title: 'Customers Transaction Analysis',
    description: 'A comprehensive database management system for a small business.',
    longDescription: "<h4>Project Overview</h4><p>This project involved designing and implementing a relational database using PostgreSQL. The system managed inventory, sales, and customer data. I also built a small web interface with Flask for data entry and reporting. The project showcases skills in database design, SQL querying, and building data-driven applications.</p>",
    images: ['/images/cowrywise.png'],
    tags: ['Database', 'SQL', 'MySQL','PostgreSQL', 'MongoDB', 'Pyspark'],
    link: '/project/database-management-system',
    aiHint: 'database servers',
    sourceCodeLink: 'https://github.com/sheunq/DataAnalytics-Assessment/tree/main',
  
    sentimentAnalysisSection: {
// Suggested code may be subject to a license. Learn more: ~LicenseLog:1192901264.
      title: 'Project Overview',
      content: "Customers who left within the last month - the column is called Churn Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies Customer account information - how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges Demographic info about customers - gender, age range, and if they have partners and dependents Inspiration To explore this type of models and learn more about the subject."
    },
    code: `
SELECT 
    owner_id,
    name,
    SUM(savings_plan) savings_count,
    SUM(Investment_plan) investment_count,
    SUM(total_deposits) total_deposits
FROM
    (SELECT 
        name,
            owner_id,
            Investment_plan,
            savings_plan,
            total_deposits
    FROM
        (SELECT 
        name,
            owner_id,
            Investment_plan,
            savings_plan,
            CASE
                WHEN investment_plan = 0 AND savings_plan = 1 THEN 'savings'
                WHEN investment_plan = 1 AND savings_plan = 0 THEN 'investment'
                ELSE 'non'
            END AS 'savings_investment_remark',
            total_deposits
    FROM
        (SELECT 
        CONCAT(customuser.first_name, ' ', customuser.last_name) AS name,
            savings.owner_id,
            plan.is_a_fund investment_plan,
            plan.is_regular_savings savings_plan,
            savings.confirmed_amount total_deposits
    FROM
        users_customuser customuser
    JOIN savings_savingsaccount savings ON customuser.id = savings.id
    JOIN plans_plan plan ON savings.owner_id = plan.owner_id) subquery1) subquery2
    WHERE
        (savings_investment_remark = 'savings'
            OR savings_investment_remark = 'investment')
            AND total_deposits != 0) subquery3
GROUP BY name , owner_id
HAVING savings_count != 0
    AND investment_count != 0
ORDER BY total_deposits;
    `

  },
  {
    id: 5,
    slug: 'Bulldozer-Sale-Price-Prediction',
    title: 'Bulldozer Sale Price Prediction',
    description: "This project tackled a real-world business problem sourced from a Kaggle competition hosted by Caterpillar Inc. Using structured data containing product specifications, usage history, and transactional attributes, I built a machine learning pipeline capable of forecasting future equipment prices a key tool for optimizing resale strategy in capital-intensive industries.",
    longDescription: `  <h2 style="color: #2c3e50;">🚜 Bulldozer Sale Price Prediction</h2>
  <p><strong>Project Type:</strong> Regression Model | <strong>Industry Focus:</strong> Heavy Equipment / Construction | <strong>Tools Used:</strong> Python, Scikit-learn, Pandas, Matplotlib, XGBoost</p>

  <h3 style="color: #34495e;">🔍 Project Overview</h3>
  <p>
    This project aims to predict the sale price of used bulldozers using historical data and machine learning regression techniques. Accurate price forecasting helps construction firms and dealers make informed purchasing and resale decisions.
  </p>

  <h3 style="color: #34495e;">🎯 Objectives</h3>
  <ul>
    <li>Develop a regression model to estimate the resale price of bulldozers</li>
    <li>Analyze the impact of machine specifications, usage, and year of manufacture on price</li>
    <li>Assist stakeholders in asset valuation and financial planning</li>
  </ul>

  <h3 style="color: #34495e;">🧩 Dataset</h3>
  <ul>
    <li><strong>Source:</strong> Blue Book for Bulldozer Sales (via Kaggle)</li>
    <li><strong>Features:</strong> Model ID, Year Made, Machine Hours, Product Size, Usage Type, Enclosure, Hydraulics, etc.</li>
    <li><strong>Target:</strong> Sale Price (USD)</li>
  </ul>

  <h3 style="color: #34495e;">🛠️ Techniques & Methodology</h3>
  <ul>
    <li>Data cleaning: handled null values, parsed date features, and removed outliers</li>
    <li>Feature engineering: extracted year sold, equipment age, and categorical encodings</li>
    <li>Applied models: Linear Regression, Random Forest Regressor, XGBoost Regressor</li>
    <li>Evaluated using RMSLE (Root Mean Squared Log Error), R² score</li>
  </ul>

  <h3 style="color: #34495e;">🚀 Key Results</h3>
  <ul>
    <li>XGBoost Regressor delivered the best performance with RMSLE under 0.25</li>
    <li>Model year, usage hours, and product size were strong predictors of price</li>
    <li>Final model generalized well on unseen data with low prediction error</li>
  </ul>

  <h3 style="color: #34495e;">💼 Business Impact</h3>
  <p>
    This pricing model can be deployed by equipment dealers and auction platforms to offer real-time price estimates, identify undervalued assets, and streamline procurement strategy—reducing overpayment risk in multi-million dollar equipment portfolios.
  </p>`,
    images: ['/images/bulldozer.png'],
    tags: ['Data-Science', 'Python', 'Scikit-learn',"Pandas","Numpy","Tensorflow","Keras","Pytorch"],
    link: '/project/customer-churn-prediction',
    sourceCodeLink: 'https://colab.research.google.com/drive/1WVol9PMr_TP3oyXUuAM2CIEku8CaF31q?usp=sharing#scrollTo=Su9JFe6BXFs9',
    aiHint: 'data analytics',
    sentimentAnalysisSection: {
// Suggested code may be subject to a license. Learn more: ~LicenseLog:1192901264.
      title: 'Project Overview',
      content: "Customers who left within the last month - the column is called Churn Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies Customer account information - how long they've been a-customer, contract, payment method, paperless billing, monthly charges, and total charges Demographic info about customers - gender, age range, and if they have partners and dependents Inspiration To explore this type of models and learn more about the subject."
    },
    code: `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (r2_score,
                             mean_absolute_error,
                             mean_squared_error)
import pickle
sns.set()

import os
image_path=os.path.join('.','images','all_image')
os.makedirs(image_path,exist_ok=True)

def savefig(fig_id,tight_layout=True,resolution=300,fig_extension='png'):
  path=os.path.join(image_path,fig_id+'.'+fig_extension)
  print(f'saving figure {fig_id}')
  if tight_layout:
    plt.tight_layout()

  plt.savefig(path,format=fig_extension,dpi=resolution)`
    
  }
  
  ,
  {
    id: 6,
    slug: 'Credit-Score-Prediction',
    title: 'Credit Score Prediction',
    description: "This project focuses on building a predictive model to estimate an individual's credit score category-poor, Standard, or Good-based on financial and behavioral attributes. The goal is to assist financial institutions in automating credit risk assessment, improving loan approval decisions, and reducing default rates.",
    longDescription: `
      <h1 style="color: #2c3e50;">📊 Credit Score Prediction</h1>
  <p><strong>Project Type:</strong> Machine Learning Model | <strong>Industry Focus:</strong> Finance | <strong>Tools Used:</strong> Python, Scikit-learn, Pandas, Matplotlib, Seaborn</p>

  <h3 style="color: #34495e;">🔍 Project Overview</h3>
  <p>
    Built a supervised learning model to predict individual credit scores—categorized as <em>Poor</em>, <em>Standard</em>, or <em>Good</em>—based on key financial and behavioral indicators. The project aims to support financial institutions in automating credit risk evaluation and improving lending decisions.
  </p>

  <h3 style="color: #34495e;">🎯 Objectives</h3>
  <ul>
    <li>Classify credit scores using supervised machine learning algorithms</li>
    <li>Analyze the relationship between financial behavior and creditworthiness</li>
    <li>Provide a data-driven foundation for credit risk assessment</li>
  </ul>

  <h3 style="color: #34495e;">🧩 Dataset</h3>
  <ul>
    <li><strong>Source:</strong> Public datasets (Kaggle / UCI repository)</li>
    <li><strong>Features:</strong> Age, Income, Loan Amount, Credit Inquiries, Payment History, Credit Utilization, etc.</li>
    <li><strong>Target:</strong> Credit Score Category (Poor, Standard, Good)</li>
  </ul>

  <h3 style="color: #34495e;">🛠️ Techniques & Methodology</h3>
  <ul>
    <li>Performed exploratory data analysis (EDA) and data cleaning</li>
    <li>Created engineered features like debt-to-income ratio and account age</li>
    <li>Applied models: Logistic Regression, Random Forest, XGBoost</li>
    <li>Evaluated performance using Accuracy, F1-Score, Confusion Matrix, ROC-AUC</li>
  </ul>

  <h3 style="color: #34495e;">🚀 Key Results</h3>
  <ul>
    <li>Best model (XGBoost) achieved ~89% accuracy on test data</li>
    <li>Feature importance showed payment history and utilization ratio as top predictors</li>
    <li>Deployed an interactive scoring tool using Streamlit (optional integration)</li>
  </ul>

  <h3 style="color: #34495e;">💼 Business Impact</h3>
  <p>
    This model demonstrates how financial service providers can leverage AI to optimize loan approvals, minimize default risk, and offer personalized financial products based on customer credit profiles.
  </p>`,
    images: ['/images/creditscore.png'],
    tags: ['Data-Science', 'Python', 'Scikit-learn',"Pandas","Numpy","Tensorflow","Keras","Pytorch"],
    link: '/project/customer-churn-prediction',
    sourceCodeLink: 'https://colab.research.google.com/drive/1nJ7UOxfwQ_eWCYc2s4QYvqbLgUHN1KUK',
    aiHint: 'data analytics',
    sentimentAnalysisSection: {
      title: 'Key Results',
      content: "Achieved ~89% accuracy with the XGBoost model. Identified that repayment history and credit utilization were the most influential features. Deployed the final model using Streamlit for an interactive credit risk scoring demo."
    },
    code: `
import pandas as pd
import numpy as np
import plotly.express as px
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer,OrdinalEncoder
from imblearn.over_sampling import SMOTE
import re

from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
BaggingClassifier,ExtraTreesClassifier,
RandomForestClassifier, StackingClassifier,
HistGradientBoostingClassifier)
from xgboost import XGBClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix, precision_score,recall_score,f1_score,roc_auc_score
import joblib

import warnings
warnings.filterwarnings('ignore')

# Load the dataset

data=pd.read_csv('/content/drive/MyDrive/Colab_Notebooks/Credit Score/train.csv')
data.sample(10)

# Preprocessing
data.select_dtypes(include='object').head()
data['Payment_of_Min_Amount']=data['Payment_of_Min_Amount'].map({'NM':'No',
                                   'No':'No',
                                   'Yes':'Yes'})
data['Payment_of_Min_Amount'].unique()
for i in data.select_dtypes(include='object').columns[:2]:
  display(data.groupby('Credit_Score')[i].value_counts().sort_index().unstack().style.background_gradient(cmap='RdBu_r',axis=0))


# Initialize and train the model
bagging = BaggingClassifier(n_jobs=-1)
extraTrees = ExtraTreesClassifier(max_depth=10, n_jobs=-1)
randomForest = RandomForestClassifier(n_jobs=-1)
histGradientBoosting = HistGradientBoostingClassifier()
XGB = XGBClassifier(n_jobs=-1)

model = StackingClassifier([
    ('bagging', bagging),
    ('extraTress', extraTrees),
    ('randomforest', randomForest),
    ('histGradientBoosting', histGradientBoosting),
    ('XGB', XGB)
    ], n_jobs=-1)

model=model.fit(x_train, y_train)
y_pred=model.predict(x_test)

# Evaluate the model
print(f'Accuracy Score: {accuracy_score(y_pred,y_test)}')
precisionscore=precision_score(y_pred,y_test,average='weighted')
f1score=f1_score(y_pred,y_test,average='weighted')
print(f'Precision Score: {precisionscore}')
print(f'f1 Score: {f1score}')
`
    
  },
  {
    id: 7,
    slug: 'customer-booking-prediction',
    title: 'Customer Booking Prediction',
    description: "This Customer Booking Prediction project uses historical transaction data, customer behavior patterns, and contextual variables to forecast the likelihood and timing of future bookings. By applying machine learning models such as Logistic Regression, XGBoost, or Neural Networks, organizations can identify which customers are most likely to make a booking, when they are likely to do so, and what factors influence their decision",
    longDescription: `
    <h2 style="color: #2c3e50;">🧳 Customer Booking Prediction</h2>
  <p><strong>Project Type:</strong> Classification Model | <strong>Industry Focus:</strong> Travel & Hospitality | <strong>Tools Used:</strong> Python, Pandas, Scikit-learn, Matplotlib, LightGBM</p>

  <h3 style="color: #34495e;">🔍 Project Overview</h3>
  <p>
    Developed a machine learning model to predict whether a customer will complete a hotel booking. The model supports marketing teams in targeting likely bookers and minimizing abandoned reservation attempts.
  </p>

  <h3 style="color: #34495e;">🎯 Objectives</h3>
  <ul>
    <li>Classify potential customers as bookers or non-bookers based on behavioral data</li>
    <li>Improve booking conversion rates through predictive insights</li>
    <li>Enable personalized marketing and dynamic pricing strategies</li>
  </ul>

  <h3 style="color: #34495e;">🧩 Dataset</h3>
  <ul>
    <li><strong>Source:</strong> Kaggle (Hotel Booking Demand Dataset)</li>
    <li><strong>Features:</strong> Booking channel, number of adults/children, lead time, room type, previous cancellations, special requests, etc.</li>
    <li><strong>Target:</strong> Booking status (1 = Booked, 0 = Not Booked)</li>
  </ul>

  <h3 style="color: #34495e;">🛠️ Techniques & Methodology</h3>
  <ul>
    <li>Performed EDA and cleaned inconsistencies (e.g., null values, outliers)</li>
    <li>Feature engineering: calculated booking window, grouped customer types</li>
    <li>Trained models: Logistic Regression, Decision Trees, LightGBM</li>
    <li>Evaluated with metrics: Accuracy, Precision, Recall, AUC-ROC</li>
  </ul>

  <h3 style="color: #34495e;">🚀 Key Results</h3>
  <ul>
    <li>LightGBM achieved the best performance with 90%+ accuracy and strong recall</li>
    <li>Lead time, number of special requests, and previous cancellations were top predictors</li>
    <li>Insights enabled segmentation of likely bookers for targeted follow-ups</li>
  </ul>

  <h3 style="color: #34495e;">💼 Business Impact</h3>
  <p>
    This solution helps hotel and travel platforms reduce cart abandonment, optimize promotional spending, and maximize revenue per user by anticipating customer behavior before conversion takes place.
  </p> 
    `,
    images: ['/images/customerbooking.png'],
    tags: ['Data-Science', 'Python', 'Scikit-learn',"Pandas","Numpy","Tensorflow","Keras","Pytorch"],
    link: '/project/customer-churn-prediction',
    sourceCodeLink: 'https://colab.research.google.com/drive/1YFXKtdgKGwFDAHvWi9AvDli4D7be58s9#scrollTo=wwLuwAIGrdeb',
    aiHint: 'data analytics',
    sentimentAnalysisSection: {
      title: 'Key Results',
      content: "Achieved ~84% r2 Score with the RandomForestRegressor with skfold model. Identified that repayment history and credit utilization were the most influential features. Deployed the final model using Streamlit for an interactive credit risk scoring demo."
    },
    code: `
import sweetviz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
sns.set()
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

# Load the dataset

df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/customer_booking.csv', sep=',' , encoding='latin-1')
df.head()

# Preprocessing
encode=OrdinalEncoder()
df[['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin']]=pd.DataFrame(encode.fit_transform(df.select_dtypes(include='object')),columns=['sales_channel', 'trip_type', 'flight_day', 'route', 'booking_origin'])


# Initialize and train the model
bagging = BaggingClassifier(n_jobs=-1)
extraTrees = ExtraTreesClassifier(max_depth=10, n_jobs=-1)
randomForest = RandomForestClassifier(n_jobs=-1)
histGradientBoosting = HistGradientBoostingClassifier()
XGB = XGBClassifier(n_jobs=-1)

model = StackingClassifier([
    ('bagging', bagging),
    ('extraTress', extraTrees),
    ('randomforest', randomForest),
    ('histGradientBoosting', histGradientBoosting),
    ('XGB', XGB)
    ], n_jobs=-1)

model=model.fit(x_train, y_train)
y_pred=model.predict(x_test)

# Evaluate the model
skf=StratifiedKFold(shuffle=True,n_splits=2, random_state=42)
for x_train_index,y_train_index in skf.split(x,y):
  x_train_fold=x_train.iloc[x_train_index]
  y_train_fold=y_train.iloc[y_train_index]
  x_test_fold=x_train.iloc[x_train_index]
  y_test_fold=y_train.iloc[y_train_index]
`
    
  },
  {
    id: 8,
    slug: 'bike-sharing-prediction',
    title: 'Bike Sharing Prediction',
    description: "This project Bike Sharing Prediction leverages historical rental data, weather patterns, seasonal trends, and time-based variables to forecast future demand for shared bicycles. This predictive modeling process typically involves data preprocessing, feature engineering, and the application of machine learning algorithms such as Random Forest and Gradient Boosting",
    longDescription: `
        <h2 style="color: #2c3e50;">🧳 Customer Booking Prediction</h2>
  <p><strong>Project Type:</strong> Classification Model | <strong>Industry Focus:</strong> Travel / Hospitality / SaaS | <strong>Tools Used:</strong> Python, Scikit-learn, Pandas, Matplotlib, Seaborn</p>

  <h3 style="color: #34495e;">🔍 Project Overview</h3>
  <p>
    Developed a predictive classification model to determine the likelihood of a customer completing a booking. The project enables businesses to reduce cart abandonment, optimize marketing strategies, and increase conversion rates.
  </p>

  <h3 style="color: #34495e;">🎯 Objectives</h3>
  <ul>
    <li>Predict whether a user will finalize a booking based on interaction data</li>
    <li>Identify key behavioral signals that drive or hinder booking decisions</li>
    <li>Support customer retention and upsell strategies with actionable insights</li>
  </ul>

  <h3 style="color: #34495e;">🧩 Dataset</h3>
  <ul>
    <li><strong>Source:</strong> Public travel/hotel booking dataset (Kaggle / simulated CRM data)</li>
    <li><strong>Features:</strong> Session duration, page views, referral source, device type, number of searches, past bookings, user demographics</li>
    <li><strong>Target:</strong> Booking Status (Booked / Not Booked)</li>
  </ul>

  <h3 style="color: #34495e;">🛠️ Techniques & Methodology</h3>
  <ul>
    <li>Exploratory data analysis and feature selection using correlation heatmaps</li>
    <li>Data preprocessing: scaling, encoding categorical variables, handling class imbalance</li>
    <li>Applied models: Logistic Regression, Random Forest, Gradient Boosting</li>
    <li>Evaluated with Precision, Recall, F1-Score, and ROC-AUC</li>
  </ul>

  <h3 style="color: #34495e;">🚀 Key Results</h3>
  <ul>
    <li>Achieved an F1-score of 0.87 with the Gradient Boosting model</li>
    <li>Top features influencing bookings: session duration, device type, referral channel</li>
    <li>Created a scoring dashboard to help sales teams prioritize high-potential leads</li>
  </ul>

  <h3 style="color: #34495e;">💼 Business Impact</h3>
  <p>
    This model empowers marketing and sales teams with predictive intelligence, allowing proactive customer engagement, reducing churn, and increasing overall booking conversion. It can be integrated into CRM workflows to score and segment users in real-time.
  </p>`,
    images: ['/images/bike sharing.png'],
    tags: ['Data-Science', 'Python', 'Scikit-learn',"Pandas","Numpy","Tensorflow","Keras","Pytorch"],
    link: '/project/customer-churn-prediction',
    sourceCodeLink: 'https://www.kaggle.com/code/seunayegboyin/bike-share-prediction',
    aiHint: 'data analytics',
    sentimentAnalysisSection: {
      title: 'Key Results',
      content: "Achieved ~99.7% r2 Score with the RandomForestRegressor with skfold model. Identified that repayment history and credit utilization were the most influential features. Deployed the final model using Streamlit for an interactive credit risk scoring demo."
    },
    code: `
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import joblib
sns.set()

# Load the dataset

df=pd.read_csv('/kaggle/input/bike-share-daily-data/bike_sharing_daily.csv')
df.head()

# Preprocessing
num_trans=Pipeline(steps=[('imputer',SimpleImputer()),('scaler',StandardScaler())])
# Categorical tranformation
cat_trans=Pipeline(steps=[('imputer',SimpleImputer(strategy='constant')),('encoder',OrdinalEncoder())])

numeric_features = ['temp', 'atemp', 'hum', 'windspeed','casual','registered']
categorical_features = ['season', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
# column Tranformer
preprocessor=ColumnTransformer(transformers=[('numeric', num_trans, numeric_features),
                                             ('categorical', cat_trans, categorical_features)])

# Initialize and train the model
model=pipeline.fit(x_train,y_train)
model.score(x_train,y_train)

# Evaluate the model
y_pred=model.predict(x_test)
pd.DataFrame({'r2_score':[r2_score(y_test,y_pred).round(3)*100],'mean_squared_error':[mean_squared_error(y_test,y_pred)],'mean_absolute_error':[mean_absolute_error(y_test,y_pred)]})
`    
  },
  {
    id: 9,
    slug: 'customer-segmentation',
    title: 'Customer Segmentation',
    description: "Developed a predictive classification model to determine the likelihood of a customer completing a booking. The project enables businesses to reduce cart abandonment, optimize marketing strategies, and increase conversion rates.",
    longDescription: `
        <h2 style="color: #2c3e50;">🧠 Customer Segmentation</h2>
  <p><strong>Project Type:</strong> Unsupervised Learning | <strong>Industry Focus:</strong> Retail / E-commerce / B2C | <strong>Tools Used:</strong> Python, Pandas, Scikit-learn, Matplotlib, Seaborn</p>

  <h3 style="color: #34495e;">🔍 Project Overview</h3>
  <p>
    Applied clustering techniques to segment customers based on purchasing behavior, demographics, and engagement patterns. This project helps businesses personalize marketing campaigns, improve customer retention, and identify high-value clients.
  </p>

  <h3 style="color: #34495e;">🎯 Objectives</h3>
  <ul>
    <li>Group customers into distinct segments based on behavior and value</li>
    <li>Enable targeted promotions, upselling strategies, and personalized experiences</li>
    <li>Provide data-driven insights to enhance customer lifetime value (CLV)</li>
  </ul>

  <h3 style="color: #34495e;">🧩 Dataset</h3>
  <ul>
    <li><strong>Source:</strong> E-commerce customer dataset (Kaggle / simulated CRM data)</li>
    <li><strong>Features:</strong> Age, Gender, Annual Income, Spending Score, Purchase Frequency, Tenure, etc.</li>
  </ul>

  <h3 style="color: #34495e;">🛠️ Techniques & Methodology</h3>
  <ul>
    <li>Data preprocessing: scaling, outlier removal, and feature selection</li>
    <li>Used K-Means clustering with the Elbow Method and Silhouette Score for optimal K</li>
    <li>Visualized clusters using PCA and 2D scatter plots</li>
    <li>Profiled each cluster based on income, spending, and engagement</li>
  </ul>

  <h3 style="color: #34495e;">🚀 Key Results</h3>
  <ul>
    <li>Identified 4 key customer segments: High Spenders, Budget Shoppers, Infrequent Buyers, and Loyal Customers</li>
    <li>Enabled precision targeting for promotions and loyalty rewards</li>
    <li>Produced visual cluster maps for marketing and sales alignment</li>
  </ul>

  <h3 style="color: #34495e;">💼 Business Impact</h3>
  <p>
    This segmentation model allows companies to tailor messaging, pricing, and product offerings to specific customer groups, leading to increased conversion rates, better customer retention, and maximized return on marketing investment (ROMI).
  </p>`,
    images: ['/images/Customer Segmentation.png'],
    tags: ['Data-Science', 'Python', 'Scikit-learn',"Pandas","Numpy","Tensorflow","Keras","Pytorch"],
    link: '/project/customer-churn-prediction',
    sourceCodeLink: 'https://www.kaggle.com/code/seunayegboyin/customer-segmentation-with-kmeans-and-pca#PCA',
    aiHint: 'data analytics',
    sentimentAnalysisSection: {
      title: '💼 Business Impact',
      content: "This segmentation model allows companies to tailor messaging, pricing, and product offerings to specific customer groups, leading to increased conversion rates, better customer retention, and maximized return on marketing investment (ROMI)"
    },
    code: `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import missingno as msno
from IPython.display import display
from sklearn.preprocessing import OrdinalEncoder
sns.set()
import warnings
warnings.filterwarnings('ignore')

# Load the dataset

train=pd.read_csv('../input/customer-segmentation/train.csv')
train[['ID', 'Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession',
       'Work_Experience', 'Spending_Score', 'Family_Size']]

# Preprocessing
num_trans=Pipeline(steps=[('imputer',SimpleImputer()),('scaler',StandardScaler())])
# Categorical tranformation
encode=OrdinalEncoder(dtype=object)
customer_encode=pd.DataFrame(encode.fit_transform(customer),columns=customer.columns)
customer_encode.head(3)

stand=StandardScaler()
stand=stand.fit_transform(customer_encode)
stand

# Initialize and train the model
wcss=[]

for i in range(1,11):
    kmean=KMeans(n_clusters=i).fit(stand)
    wcss.append(kmean.inertia_)
print(wcss)

# Evaluate the model
y_pred=model.predict(x_test)
pd.DataFrame({'r2_score':[r2_score(y_test,y_pred).round(3)*100],'mean_squared_error':[mean_squared_error(y_test,y_pred)],'mean_absolute_error':[mean_absolute_error(y_test,y_pred)]})
`    
  },
  {
    id: 10,
    slug: 'sales-dashboard-using-power-BI',
    title: 'Sales Dashboard using Power BI',
    description: 'This Sales Performance Dashboard provides actionable insights into product sales, customer segmentation, and regional performance. Built with Power BI, this dashboard supports strategic decision-making by consolidating large volumes of transactional data into an interactive and visually intuitive interface.',
    longDescription: `<h1>📊 Sales Dashboard using Power BI</h1>

  <div class="section">
    <h2>🚀 Dashboard Breakdown</h2>
    <p><span class="highlight">Total Quantity Sold:</span> 49K units</p>

    <h3>Top Customer Segments</h3>
    <ul>
      <li>🥇 <strong>Gold:</strong> 17K</li>
      <li>🥈 <strong>Silver:</strong> 17K</li>
      <li>🥉 <strong>Bronze:</strong> 15K</li>
    </ul>

    <p><span class="highlight">Highest Selling Product Category:</span> Electronics (34%)</p>
    <p><span class="highlight">Top Performing Subcategory:</span> Headphones (24.16%)</p>
    <p><span class="highlight">Best Performing States:</span> Armed Forces (various), Montana, Nevada, Wisconsin (8% each)</p>
  </div>

  <div class="section">
    <h2>📊 Segment-Level Insights</h2>

    <h3>🥇 Gold Customers</h3>
    <ul>
      <li><strong>Total Sales:</strong> 17K</li>
      <li><strong>Monthly Trend:</strong> Highs in Jan (1.8K) and Jul (1.7K); dip in Jun (0.9K)</li>
      <li><strong>Stability:</strong> Volatile with Nov recovery (1.5K)</li>
    </ul>

    <h3>🥈 Silver Customers</h3>
    <ul>
      <li><strong>Total Sales:</strong> 17K</li>
      <li><strong>Monthly Trend:</strong> Steady growth, peak in Sep (2.1K)</li>
      <li><strong>Stability:</strong> Consistent</li>
    </ul>

    <h3>🥉 Bronze Customers</h3>
    <ul>
      <li><strong>Total Sales:</strong> 15K</li>
      <li><strong>Monthly Trend:</strong> Weak Mar–May; peaks in Jul (1.7K) and Sep (1.2K)</li>
      <li><strong>Stability:</strong> Most volatile segment</li>
    </ul>
  </div>

  <div class="section">
    <h2>🛍️ Product-Level Insights</h2>

    <h3>🔧 Subcategory Performance</h3>
    <ul>
      <li>Headphones – 24.16%</li>
      <li>Academic – 14.4%</li>
      <li>Shirt – 11.4%</li>
      <li>Action Dolls – 8.9%</li>
      <li>Gardening – 8.4%</li>
    </ul>

    <div class="recommendation">
      ✅ <strong>Recommendation:</strong> Focus marketing and inventory on Headphones, Academic materials, and Apparel.
    </div>

    <h3>🧩 Product Categories</h3>
    <ul>
      <li>Electronics: 17K units (34%)</li>
      <li>Books: 13K units (25.29%)</li>
      <li>Toys: 9K units (17.2%)</li>
      <li>Home & Garden: 7K units (13.52%)</li>
      <li>Clothing: 5K units (9.94%)</li>
    </ul>

    <div class="recommendation">
      📦 <strong>Inventory Planning:</strong> Prioritize Electronics and Books in procurement and promotion.
    </div>
  </div>

  <div class="section">
    <h2>🌍 Geographic Distribution</h2>
    <ul>
      <li><strong>Top Contributor:</strong> Armed Forces regions – 15%, 13%, 12%</li>
      <li><strong>Other Strong States:</strong> Montana, Nevada, Wisconsin – 8% each</li>
      <li><strong>Moderate States:</strong> Alaska, North Dakota, Washington – 7% each</li>
    </ul>

    <div class="recommendation">
      🌐 <strong>Insight:</strong> Contract-based demand likely in Armed Forces regions. Consider niche targeting and fulfillment strategies.
    </div>
  </div>

  <div class="section">
    <h2>📈 Trends & Red Flags</h2>

    <div class="alert">
      📉 <strong>Underperformance Alert:</strong> Bronze segment is flat across months. Subcategories like Decor, Camera, and Furniture contribute less than 5%.
    </div>

    <div class="recommendation">
      🛠 <strong>Optimization Opportunities:</strong><br>
      • Personalize promotions for Bronze customers<br>
      • Use geo-marketing to penetrate under-7% states
    </div>
  </div>

  <div class="section">
    <h2>💡 Strategic Recommendations</h2>
    <ul>
      <li><strong>Amplify High-Performers:</strong> Push electronics and headphones during peak months (Jan, Sept)</li>
      <li><strong>Targeted Retention:</strong> Loyalty offers for Bronze customers</li>
      <li><strong>Stock Reallocation:</strong> Reduce low-performing categories</li>
      <li><strong>Forecasting:</strong> Prepare for future spikes based on past peaks</li>
    </ul>
  </div>
`,
    images: [
     '/images/sold2023.png',
      '/images/sold2022.png',
      '/images/sold2021.png',
      '/images/sold2021.png',

    ],
    tags: ['Visualization', 'PowerBI'],
    link: '/project/sales-forecasting',
    aiHint: 'financial chart',
  },
  {
    id: 11,
    slug: 'vehicle-repair-dashboard',
    title: 'Vehicle Repair Dashboard',
    description: 'This Vehicle Repair Dashboard provides a strategic lens into fleet maintenance activities across suppliers, vehicle brands, and service types. It enables data-driven decision-making by highlighting repair patterns, supplier performance, and maintenance trends.',
    longDescription: `
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Vehicle Service & Repair Dashboard Analysis</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      color: #333;
      padding: 20px;
    }
    h1, h2, h3 {
      color: #003366;
    }
    .section {
      margin-bottom: 30px;
    }
    .highlight {
      font-weight: bold;
      color: #0077b6;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    table, th, td {
      border: 1px solid #ccc;
    }
    th, td {
      padding: 10px;
      text-align: left;
    }
    .emoji {
      font-size: 1.1em;
    }
  </style>
</head>
<body>

  <h1>🚗 Vehicle Service & Repair Dashboard Analysis</h1>

  <div class="section">
    <h2 class="emoji">🔧 1. Workshop/Supplier Performance</h2>
    <p><span class="highlight">Top Performer:</span> <strong>Mecho Auto Tech</strong> handled <strong>43 repairs/services (43.43%)</strong>, leading the supplier chart. Dominated with <strong>30 Routine</strong> and <strong>14 Repair</strong> services.</p>
    <p><span class="highlight">Runner-Up:</span> <strong>Adhoc (Others)</strong> followed closely with <strong>34 services</strong>, mostly <strong>Routine (28)</strong>.</p>
    <p><span class="highlight">Low-Activity Vendors:</span> Mikano, Elizade (JAC), and Affordable Cars — only 1 record each.</p>
    <p><span class="highlight">Insight:</span> Heavy reliance on 2 providers. Diversify to mitigate vendor concentration risk.</p>
  </div>

  <div class="section">
    <h2 class="emoji">🚘 2. Vehicle Repair/Service Insights</h2>
    <p><span class="highlight">Service Types:</span></p>
    <ul>
      <li><strong>Repair:</strong> 73.74%</li>
      <li><strong>Routine Service:</strong> 25.25%</li>
      <li><strong>Routine:</strong> 1.01% (negligible)</li>
    </ul>
    <p><span class="highlight">Top Brands Requiring Repairs:</span></p>
    <ul>
      <li><strong>JAC:</strong> 38 cases, mostly serviced by 5Speed Tech.</li>
      <li><strong>Mini Van:</strong> 18 repairs — serviced by Mecho Auto Tech.</li>
      <li><strong>Nissan Models:</strong> Almera, Pick Up, and Kicks show consistent issues.</li>
    </ul>
    <p><span class="highlight">Observation:</span> JAC repairs heavily rely on a single workshop — vendor dependency risk.</p>
  </div>

  <div class="section">
    <h2 class="emoji">📊 3. Service Trends in 2024</h2>
    <p><span class="highlight">Repair Trend:</span> Noticeable spike mid-year (July–August).</p>
    <p><span class="highlight">Routine Services:</span> Fluctuate at low levels — preventive maintenance underutilized.</p>
    <p><span class="highlight">Operational Risk:</span> Reactive maintenance dominates. This is inefficient and cost-intensive.</p>
  </div>

  <div class="section">
    <h2 class="emoji">📌 Strategic Recommendations</h2>
    <ul>
      <li><strong>Boost Preventive Maintenance:</strong> Increase Routine/Routine Service activities to avoid unplanned repairs.</li>
      <li><strong>Balance Vendor Load:</strong> Reduce dependency on Mecho Auto Tech and Adhoc. Onboard and empower alternatives.</li>
      <li><strong>JAC & Mini Van Analysis:</strong> High repair frequency — investigate usage patterns or quality issues.</li>
      <li><strong>Monitor Monthly Trends:</strong> Use monthly data to forecast and reduce peak maintenance pressure.</li>
    </ul>
  </div>

  <div class="section">
    <h2 class="emoji">🧠 Executive Summary</h2>
    <table>
      <thead>
        <tr>
          <th>Metric</th>
          <th>Top Performer / Insight</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>🛠️ Most Used Workshop</td>
          <td>Mecho Auto Tech (43 services)</td>
        </tr>
        <tr>
          <td>🚗 Most Serviced Vehicle</td>
          <td>JAC (38 repairs, serviced by 5Speed Tech.)</td>
        </tr>
        <tr>
          <td>🔧 Dominant Service Type</td>
          <td>Repairs (73%) – preventive services are lagging</td>
        </tr>
        <tr>
          <td>📉 Underutilized Vendors</td>
          <td>Affordable Cars, Mikano, Elizade</td>
        </tr>
        <tr>
          <td>📈 Trend Observation</td>
          <td>Repair peaks mid-year; preventive efforts are minimal</td>
        </tr>
      </tbody>
    </table>
  </div>

</body>
</html>

`,
    images: [
            '/images/repair2.png',
      '/images/repair1.png',
      '/images/repair3.png',
    ],
    tags: ['Visualization', 'PowerBI'],
    link: '/project/sales-forecasting',
    aiHint: 'financial chart',
  },

  {
    id: 12,
    slug: 'Web-Scraping-and-Image-Classification-with-TensorFlow',
    title: 'Web Scraping and Image Classification with TensorFlow',
    description: 'This project demonstrates my ability to combine web scraping techniques with machine learning to classify images. The goal was to build a pipeline that scrapes images of shoes from a website and uses them for training a machine learning model.',
    longDescription: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Scraping and Image Classification with TensorFlow</title>
</head>
<body>
    <h2>Web Scraping and Image Classification with TensorFlow</h2>
    <p>This project demonstrates my ability to combine web scraping techniques with machine learning to classify images. The goal was to build a pipeline that scrapes images of shoes from a website and uses them for training a machine learning model.</p>
    
    <h3>Key Features:</h3>
    <ul>
        <li><strong>Web Scraping:</strong> I utilized <code>BeautifulSoup</code> and <code>requests</code> to scrape images from an e-commerce site, focusing on categories like formal shoes, sneakers, and boots. The images were automatically downloaded and saved into organized directories.</li>
        <li><strong>Image Preprocessing:</strong> The images were stored in separate folders for training and testing. This organization sets the foundation for an image classification task.</li>
        <li><strong>Model Setup:</strong> While the project focuses on scraping, the next logical step (which can be explored further) is to use <code>TensorFlow</code> and <code>Keras</code> to create a Convolutional Neural Network (CNN) model to classify these images into categories such as sneakers, formal shoes, and boots.</li>
        <li><strong>Scalability:</strong> The design of the scraping function allows it to easily scale for other shoe categories or even other types of products. Simply by adjusting the URLs and folder names, new categories can be added to the pipeline.</li>
    </ul>
    
    <h3>Technologies Used:</h3>
    <ul>
        <li><strong>Python:</strong> The core language for web scraping and model building.</li>
        <li><strong>TensorFlow & Keras:</strong> For machine learning model development.</li>
        <li><strong>BeautifulSoup & Requests:</strong> For scraping data from the web.</li>
        <li><strong>PIL & Matplotlib:</strong> For handling images and visualizing results.</li>
    </ul>
    
    <h3>Why This Project?</h3>
    <p>This project not only helped me refine my web scraping skills but also gave me hands-on experience in integrating web data with machine learning models. It's an excellent example of how data from websites can be harvested and used in AI-driven applications, such as in e-commerce for product categorization.</p>
</body>
</html>
`,
    images: ['/images/shoecollection.png'],
    tags: ['WebScraping', 'BeautifulSoup', 'Selenium', 'Scrapy'],
    link: '/project/sentiment-analysis-reviews',
    aiHint: 'cinema film',
    sourceCodeLink: 'https://colab.research.google.com/drive/1DUvQUFxeGz0DP7ZffDQLt4ig8c7Kf09E?usp=drive_open#scrollTo=L0v3mh51TCpD',
    sentimentAnalysisSection: {
      title: '💼 Business Impact',
      content: `This project has the potential to significantly enhance business operations, especially for e-commerce platforms. By automating the categorization and classification of product images, businesses can save time and resources that would otherwise be spent on manual labor. Here are some key business impacts:

      \n
Improved Product Categorization: Automated image classification ensures accurate categorization of products, which improves the shopping experience for customers. This makes it easier for businesses to organize their inventory and for customers to find what they are looking for quickly.
\n
Enhanced Customer Experience: By having accurate and consistent product categories, customers can easily navigate through various types of shoes (e.g., sneakers, formal shoes, boots), leading to improved satisfaction and higher conversion rates.`

    },
    code: `

    import requests
from bs4 import BeautifulSoup
from PIL import Image
from IPython.display import display
import re
import os
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow as tf

# Load Data

def download_image(train_or_test,folder_name):
  all_url=[]
  res=requests.get(f'https://shoeplace.ng/10-formal-shoes').content
  be=BeautifulSoup(res,'lxml')
  for i in range(len(be.find_all('img'))):
    try:
      all_url.append(be.find_all('img')[i]['data-full-size-image-url'])
    except:
      pass


  for i in range(len(all_url)):
    r = requests.get(all_url[i])

    with open(f"shoes/{train_or_test}/{folder_name}/{i}.jpg",'wb') as f:

      f.write(r.content)
  return

  def download_image(start_page,end_page,item_name,folder_name,train_or_test):
  all_url=[]
  for i in range(start_page,end_page):
    res=requests.get(f'https://en.afew-store.com/collections/{item_name}?offset={i}').content
    be=BeautifulSoup(res,'lxml')

    for i in range(len(be.find_all('img'))):
      try:
        all_url.append(be.find_all('img')[i]['data-src'])
      except:
        pass


# webscraping
  for i in range(len(all_url)):
    r = requests.get('https:'+all_url[i])

    with open(f"shoes/{train_or_test}/{folder_name}/{i}.jpg",'wb') as f:

      f.write(r.content)
  return

# TensorFlow
  train=keras.utils.image_dataset_from_directory('/content/shoes/train',seed=seed,image_size=(180,180),validation_split=0.2,subset='training')
val=keras.utils.image_dataset_from_directory('/content/shoes/train',seed=seed,image_size=(180,180),validation_split=0.2,subset='validation')

model=keras.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(60,5,activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(80,6,activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(100,8,activation='relu'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(6)
])

` 
  },
  {
    id: 13,
    slug: 'British-Airways-Customer-Review-Web-Scraping-Text-Classification-Sentiment-Analysis-and-Customer-Satisfaction',
    title: 'British Airways Customer Review Web Scraping, Text Classification, Sentiment Analysis, and Customer Satisfaction',
    description: 'This project demonstrates my ability to combine web scraping techniques with machine learning to classify images. The goal was to build a pipeline that scrapes images of shoes from a website and uses them for training a machine learning model.',
    longDescription: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Web Scraping, Text Classification, Sentiment Analysis, and Customer Satisfaction</title>
</head>
<body>
    <h2>Web Scraping, Text Classification, Sentiment Analysis, and Customer Satisfaction</h2>
    <p>This project combines web scraping, text classification, and sentiment analysis to evaluate customer satisfaction based on reviews or other text data. The project is built around extracting relevant data from airline websites and analyzing customer sentiments to classify whether the sentiment is positive, neutral, or negative.</p>
    
    <h3>Key Features:</h3>
    <ul>
        <li><strong>Web Scraping:</strong> Using <code>BeautifulSoup</code> and <code>Requests</code>, I scraped customer reviews and other relevant data from various airline websites (such as British Airways, Air India, Emirates, and more) to gather a rich dataset for analysis.</li>
        <li><strong>Sentiment Analysis:</strong> With sentiment analysis techniques, the project identifies and classifies customer sentiments from the scraped text. This can be used to analyze feedback, detect issues, and measure customer satisfaction.</li>
        <li><strong>Text Classification:</strong> The project also focuses on binary classification, where the system identifies positive and negative customer feedback based on review content.</li>
        <li><strong>Model Development:</strong> I used <code>TensorFlow</code> and other tools like <code>Pandas</code>, <code>NumPy</code>, and <code>Regex</code> to preprocess the data, train models, and evaluate the performance of the sentiment classifier.</li>
    </ul>
    
    <h3>Technologies Used:</h3>
    <ul>
        <li><strong>Python:</strong> Core language for web scraping, data analysis, and machine learning tasks.</li>
        <li><strong>BeautifulSoup & Requests:</strong> For scraping data from web pages.</li>
        <li><strong>TensorFlow:</strong> For creating and training machine learning models to perform sentiment analysis and classification.</li>
        <li><strong>Pandas & NumPy:</strong> For handling and manipulating datasets.</li>
        <li><strong>Regex:</strong> For text pattern matching and cleaning the scraped data.</li>
    </ul>
    
    <h3>Business Impact:</h3>
    <p>This project offers a practical solution for businesses, especially in the airline industry, to better understand customer sentiment and improve their service offerings. By analyzing customer feedback, businesses can:</p>
    <ul>
        <li>Quickly identify areas for improvement.</li>
        <li>Measure the success of new services or features.</li>
        <li>Enhance customer satisfaction and loyalty by responding to customer concerns and recognizing positive feedback.</li>
    </ul>
    <p>The ability to analyze customer sentiment on a large scale, across multiple platforms, can provide actionable insights that help businesses make data-driven decisions to enhance the customer experience.</p>
</body>
</html>

`,
    images: ['/images/customer-reviews.png'],
    tags: ['WebScraping', 'BeautifulSoup', 'Selenium', 'Scrapy'],
    link: '/project/sentiment-analysis-reviews',
    aiHint: 'cinema film',
    sourceCodeLink: 'https://colab.research.google.com/drive/1HyEpph91SLyGzCP3BX5iZi8L7Q4TRRK4?usp=drive_open#scrollTo=qpwXYULML0_k',
    sentimentAnalysisSection: {
      title: '💼 Business Impact',
      content: `This project offers a practical solution for businesses, especially in the airline industry, to better understand customer sentiment and improve their service offerings. By analyzing customer feedback, businesses can:

Quickly identify areas for improvement.
Measure the success of new services or features.
Enhance customer satisfaction and loyalty by responding to customer concerns and recognizing positive feedback.
The ability to analyze customer sentiment on a large scale, across multiple platforms, can provide actionable insights that help businesses make data-driven decisions to enhance the customer experience.`

    },
    code: `

import requests
from bs4 import BeautifulSoup as bs
import re
from IPython.display import display
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers
import os
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data/webscraping

def text_scapping(start_page,end_page,recommended_or_not_recommended,airline):
  recommended=[]
  not_recommended=[]
  for i in range(start_page,end_page):
    url=f'https://www.airlinequality.com/airline-reviews/{airline}/page/{i}/'
    re=requests.get(url).content
    result=bs(re,'lxml')

    for i in range(len(result.find_all('div',class_='text_content'))):
      import re
      if result.find_all('td',class_=re.compile('^review-value rating-'))[i].string=='no':
        not_recommended.append(re.split(r'[|]',result.find_all('div',class_='text_content')[i].text,maxsplit=2)[-1])
      elif result.find_all('td',class_=re.compile('^review-value rating-'))[i].string=='yes':
        recommended.append(re.split(r'[|]',result.find_all('div',class_='text_content')[i].text,maxsplit=2)[-1])
      else:
        pass

  if 'recommended'==recommended_or_not_recommended:
    recom_or_not_recom=recommended
  else:
    recom_or_not_recom=not_recommended
  return recom_or_not_recom


def download_text(train_or_test,recommended_or_not_recommended,good_or_bad):
  z=[0]
  for d in (os.listdir(f'/content/drive/MyDrive/Colab_Notebooks/airline/{train_or_test}/{good_or_bad}')):
      z.append(int(d.split('.')[0]))
  if 'not_recommended'==recommended_or_not_recommended:
    for k in range(len(not_recommended)):
        with open(f'/content/drive/MyDrive/Colab_Notebooks/airline/{train_or_test}/{good_or_bad}/{k+np.max(z)}.txt', 'w') as f:
          f.write(not_recommended[k])
  if 'recommended'==recommended_or_not_recommended:
    for k in range(len(recommended)):
        with open(f'/content/drive/MyDrive/Colab_Notebooks/airline/{train_or_test}/{good_or_bad}/{k+np.max(z)}.txt', 'w') as f:
          f.write(recommended[k])
  return



# TensorFlow
  def plot_text(train_or_test,good_or_bad,text_number):
  c=[]
  with open(f'/content/drive/MyDrive/Colab_Notebooks/airline/{train_or_test}/{good_or_bad}/{text_number}.txt') as r:
    c.append(r.read())

  text_len=[]
  for k in c:
    text=k.split()
  for d in text:
    text_len.append(len(d))

  letter=[]
  letter_count=[]
  for c,d in sorted(zip(text_len,text),reverse=True):
    letter_count.append(c)
    letter.append(d)


  plt.figure(figsize=(20,10))
  sns.barplot(x=letter,y=letter_count)
  plt.xticks(ticks=range(len(letter_count)),labels=letter,rotation=90)
  plt.title('distribution of n-grams')
  plt.xlabel('n-grams')
  plt.ylabel('n-grams length')



  unique_text,unique_count=np.unique(text,return_counts=True)

  unique_text_=[]
  unique_text_count=[]
  for k,i in sorted(zip(unique_count,unique_text)):
    unique_text_.append(i)
    unique_text_count.append(k)

  plt.figure(figsize=(20,10))
  sns.barplot(x=unique_text_,y=unique_text_count)
  plt.xticks(ticks=range(len(unique_text_count)),labels=unique_text_,rotation=90)
  plt.title('Frequency distribution of n-grams')
  plt.xlabel('n-grams')
  plt.ylabel('Frequencies')

  return plt.show()

  #Model Training
  model.compile(optimizer='adam',metrics=tf.metrics.BinaryAccuracy(threshold=0.0),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

` 
  },

  {
    id: 14,
    slug: 'british-airway-webscraping-with-beautifulsoup',
    title: 'British Airway Webscraping with Beautifulsoup',
    description: "This project demonstrates how web scraping can be used to extract valuable data from airline websites, specifically British Airways, using Python's BeautifulSoup library. The goal was to scrape flight-related data, such as flight details, prices, and other useful information, from the British Airways website for analysis.",
    longDescription: `
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>British Airways — Customer Review Web Scraper</title>
  <meta name="description" content="Python web scraper using BeautifulSoup to collect, clean and visualize British Airways customer reviews from AirlineQuality.com." />
  <style>
    :root{
      --bg:#ffffff; --card:#f7f9fb; --muted:#6b7280; --accent:#0b5cff; --mono: 'Courier New', monospace;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    body{background:var(--bg); color:#0f172a; margin:0; padding:40px; line-height:1.5;}
    .container{max-width:900px;margin:0 auto;}
    header{display:flex;align-items:center;gap:18px;margin-bottom:18px;}
    .logo{width:64px;height:64px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#8aa6ff);display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-family:var(--mono);}
    h1{margin:0;font-size:26px;}
    p.lead{margin:8px 0 20px;color:var(--muted);}
    .card{background:var(--card);border-radius:12px;padding:18px;margin-bottom:14px;box-shadow:0 6px 20px rgba(11,92,255,0.05);}
    h2{font-size:18px;margin:0 0 8px;}
    ul{margin:0 0 12px 20px;}
    .meta{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:14px;}
    .cta{display:inline-block;margin-top:12px;padding:10px 14px;border-radius:8px;background:var(--accent);color:#fff;text-decoration:none;font-weight:600;}
    code{background:#eef2ff;padding:4px 6px;border-radius:6px;font-family:var(--mono);font-size:13px;}
    .two-col{display:grid;grid-template-columns:1fr 320px;gap:16px;}
    .sidebar{font-size:14px;color:var(--muted);}
    footer{margin-top:26px;color:var(--muted);font-size:13px;}
    .badge{display:inline-block;background:#e8eefc;color:#0b5cff;padding:6px 8px;border-radius:999px;font-weight:600;font-size:12px;}
    @media (max-width:820px){.two-col{grid-template-columns:1fr;}.logo{width:56px;height:56px}}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">BA</div>
      <div>
        <h1>British Airways — Customer Review Web Scraper</h1>
        <p class="lead">Python pipeline that systematically collects, cleans, and visualizes passenger reviews from AirlineQuality.com — turning raw web content into operational insights.</p>
        <div class="meta">
          <span class="badge">Portfolio Project</span>
          <span>Jupyter Notebook • BeautifulSoup</span>
          <span>CSV output • Visualizations</span>
        </div>
      </div>
    </header>

    <section class="card two-col" aria-labelledby="overview">
      <div>
        <h2 id="overview">Overview</h2>
        <p>Built a robust, reusable scraper using <code>requests</code> + <code>BeautifulSoup</code> to extract British Airways customer reviews from AirlineQuality.com. The pipeline focuses on repeatable page iteration, defensive parsing, and producing clean tabular outputs for analysis.</p>

        <h2>Key Features</h2>
        <ul>
          <li><strong>Automated multi-page scraping:</strong> loops through review pages and collects structured fields (title, body, rating, date, route, aircraft details).</li>
          <li><strong>Resilient parsing logic:</strong> handles missing fields, normalizes date formats, strips HTML noise, and removes duplicate entries.</li>
          <li><strong>Preprocessing & EDA:</strong> cleans text, token counts, sentiment-ready outputs, and visual EDA using <code>pandas</code> and <code>matplotlib</code>.</li>
          <li><strong>Export-ready data:</strong> store results as compressed CSV for downstream analytics or ML pipelines.</li>
          <li><strong>Reproducible Notebook:</strong> documented steps and modular functions to enable reuse and extension.</li>
        </ul>

        <h2>Tech Stack</h2>
        <ul>
          <li>Python (requests, BeautifulSoup4)</li>
          <li>pandas, NumPy — data manipulation</li>
          <li>matplotlib, seaborn — visualization & EDA</li>
          <li>Jupyter Notebook — interactive analysis & documentation</li>
        </ul>

        <h2>Impact & Use Cases</h2>
        <p>Transforms unstructured review text into structured datasets that product, ops, and customer experience teams can query to identify recurring issues (e.g., delays, cabin service, baggage handling). Ideal for building sentiment models, trend monitoring dashboards, and evidence-based operational improvements.</p>
      </div>

      <aside class="sidebar card" aria-labelledby="run">
        <h2 id="run">How to run</h2>
        <ol>
          <li>Clone the repo and open the notebook in Jupyter.</li>
          <li>Install dependencies: <code>pip install -r requirements.txt</code> (requirements in notebook header).</li>
          <li>Run notebook cells top-to-bottom. The scraper saves results to <code>data/british_airways_reviews.csv</code>.</li>
        </ol>

        <h2>Safety & Ethics</h2>
        <p>Respect site <strong>robots.txt</strong> and rate limits. The notebook includes a polite delay between requests and a user-agent header; adapt further for production environments and legal compliance.</p>

        <h2>Files included</h2>
        <ul>
          <li><code>british_airway_webscraping_with_beautifulsoup.ipynb</code> — notebook</li>
          <li><code>requirements.txt</code> — dependencies</li>
          <li><code>data/british_airways_reviews.csv</code> — sample export (generated)</li>
        </ul>

        <a class="cta" href="#" title="Link to repository or notebook">View Notebook / Repo</a>
      </aside>
    </section>

    <section class="card" aria-labelledby="notes">
      <h2 id="notes">Notes & Next Steps</h2>
      <ul>
        <li>Convert to a modular Python package for scheduled scraping (Airflow/Cron) and add automated testing for parsing edge-cases.</li>
        <li>Enhance with NLP: sentiment analysis, topic modeling (LDA), and named-entity extraction to prioritize issues.</li>
        <li>Consider storing results in a small analytical DB (SQLite/Postgres) or pushing to an analytics dashboard for real-time monitoring.</li>
      </ul>
    </section>

    <footer>
      <div><strong>Author:</strong> Seun — Supply Chain / Data Practitioner</div>
      <div><strong>License:</strong> MIT — feel free to reuse and adapt; attribute where possible.</div>
    </footer>
  </div>
</body>
</html>


`,
    images: ['/images/britishairways.png'],
    tags: ['WebScraping', 'BeautifulSoup', 'Selenium', 'Scrapy'],
    link: '/project/sentiment-analysis-reviews',
    aiHint: 'cinema film',
    sourceCodeLink: 'https://colab.research.google.com/drive/1cEwXjgRlLRQEO9uv_vCBR9skUF0ma8Q9?usp=drive_open#scrollTo=zkhcGTrMtoJP',
    sentimentAnalysisSection: {
      title: '💼 Business Impact',
      content: `This project has the potential to add value in various business scenarios, especially in travel and airline industries:

Market Analysis: Regular scraping of flight data allows businesses to track competitive pricing and service offerings over time.

Customer Insights: Analyzing customer reviews or pricing trends can help companies adjust their strategies to meet customer expectations.

Automation: Automating the scraping process provides businesses with up-to-date data without the need for manual data collection, saving both time and resources.`

    },
    code: `

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

image_path=os.path.join('.','images','all_image')
os.makedirs(image_path,exist_ok=True)

def savefig(fig_id,tight_layout=True,resolution=300,fig_extension='png'):
    path=os.path.join(image_path,fig_id+'.'+fig_extension)
    print(f'saving figure {fig_id}')
    if tight_layout:
        plt.tight_layout()

    plt.savefig(path,format=fig_extension,dpi=resolution)

from IPython.display import display
sns.set()

# Load Data/webscraping

Route=[]
for i in range(1,100):
    url=f"https://www.airlinequality.com/airline-reviews/british-airways/page/{i}/"
    doc=requests.get(url).text
    result=BeautifulSoup(doc,'lxml')
    for i in range(0,len(result.find_all('td',class_='review-value'))):
        rv=result.find_all('td',class_='review-value')[i].string
        if (re.findall('to',rv)==['to']):
            Route1=(result.find_all('td',class_='review-value')[i].string)
            Route.append(Route1)


rating=[]

for i in range(1,100):
    url=f"https://www.airlinequality.com/airline-reviews/british-airways/page/{i}/"
    doc=requests.get(url).text
    result=BeautifulSoup(doc,'lxml')
    for i in range(0,len(result.div.find_all('time',itemprop="datePublished"))):
            date=result.div.find_all('time',itemprop="datePublished")[i].string
            name=result.div.find_all('span',itemprop="name")[i].string
            country=result.div.find_all('h3')[i].get_text(strip=True)
            comment=result.div.find_all('h2')[i+1].string

            rv=result.find_all('td',class_='review-value')[i].string
            if (re.findall('Leisure$',rv)==['Leisure'])|(re.findall('Business',rv)==['Business']):
                Type_Of_Traveller=(result.find_all('td',class_='review-value')[i].string)

            rating.append([date,name,country,comment,Type_Of_Traveller])


# Data Preprocessing

ba1=British_Airway['country'].str.split(')',n=1,expand=True)
British_Airway['country']=ba1[0].str.split('(',n=1,expand=True)[1]
British_Airway.head()

# Data Visualization
# Cabin Staff Service rating
display(rating_by_Seat_Type(groupby_Seat_Type['Cabin_Staff_Service_rating'],'Cabin_Staff_Service_rating_by_Seat_Type','Cabin Staff Service rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Food_and_Beverages_rating'],'Food_and_Beverages_rating_by_Seat_Type','Food and Beverages rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Ground_Service_rating'],'Ground_Service_rating_by_Seat_Type','Ground Service rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Inflight_Entertainment'],'Inflight_Entertainment_by_Seat_Type','Inflight Entertainment'))
display(rating_by_Seat_Type(groupby_Seat_Type['Seat_comfort_rating'],'Seat_comfort_rating_by_Seat_Type','Seat comfort rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Value_For_Money_rating'],'Value_For_Money_rating_by_Seat_Type','Value For Money rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Wifi_and_Connectivity_rating'],'Wifi_and_Connectivity_rating_by_Seat_Type','Wifi and Connectivity rating'))
 
` 
  },
  {
    id: 15,
    slug: 'ecommerce-website-webscraping',
    title: 'Ecommerce Website Webscraping',
    description: "This project uses requests and BeautifulSoup to scrape product listings from StockX’s e-commerce platform. The scraper extracts product name, description, price, and image link across multiple paginated results, storing them in a structured dataset for analysis or integration into business workflows.",
    longDescription: `
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>E-commerce Product Data Scraper (StockX)</title>
  <meta name="description" content="Python web scraper using BeautifulSoup to collect, clean, and analyze product listings from the StockX e-commerce platform." />
  <style>
    :root {
      --bg: #ffffff;
      --card: #f7f9fb;
      --muted: #6b7280;
      --accent: #0b5cff;
      --mono: 'Courier New', monospace;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    body {
      background: var(--bg);
      color: #0f172a;
      margin: 0;
      padding: 40px;
      line-height: 1.5;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
    }
    header {
      display: flex;
      align-items: center;
      gap: 18px;
      margin-bottom: 18px;
    }
    .logo {
      width: 64px;
      height: 64px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), #8aa6ff);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-weight: 700;
      font-family: var(--mono);
    }
    h1 {
      margin: 0;
      font-size: 26px;
    }
    p.lead {
      margin: 8px 0 20px;
      color: var(--muted);
    }
    .card {
      background: var(--card);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 14px;
      box-shadow: 0 6px 20px rgba(11,92,255,0.05);
    }
    h2 {
      font-size: 18px;
      margin: 0 0 8px;
    }
    ul {
      margin: 0 0 12px 20px;
    }
    .meta {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 14px;
    }
    .cta {
      display: inline-block;
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      text-decoration: none;
      font-weight: 600;
    }
    code {
      background: #eef2ff;
      padding: 4px 6px;
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 13px;
    }
    footer {
      margin-top: 26px;
      color: var(--muted);
      font-size: 13px;
    }
    .badge {
      display: inline-block;
      background: #e8eefc;
      color: #0b5cff;
      padding: 6px 8px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">EC</div>
      <div>
        <h1>E-commerce Product Data Scraper (StockX)</h1>
        <p class="lead">Python-powered web scraper that collects product details across multiple pages from StockX, enabling price monitoring, catalog building, and trend analysis.</p>
        <div class="meta">
          <span class="badge">Portfolio Project</span>
          <span>BeautifulSoup • Requests • Pandas</span>
          <span>Data Extraction • Analytics</span>
        </div>
      </div>
    </header>

    <section class="card">
      <h2>Overview</h2>
      <p>This project uses <code>requests</code> and <code>BeautifulSoup</code> to scrape product listings from StockX’s e-commerce platform. The scraper extracts product name, description, price, and image link across multiple paginated results, storing them in a structured dataset for analysis or integration into business workflows.</p>
    </section>

    <section class="card">
      <h2>Key Features</h2>
      <ul>
        <li><strong>Automated Multi-Page Scraping:</strong> Iterates through multiple pages to collect complete datasets.</li>
        <li><strong>Dynamic Data Extraction:</strong> Captures product title, description, price, and image URL.</li>
        <li><strong>Structured Storage:</strong> Saves results in Pandas DataFrame and exports to CSV.</li>
        <li><strong>Custom Function:</strong> <code>scrap(product_cat)</code> allows easy scraping of any category.</li>
        <li><strong>Error Handling:</strong> Gracefully skips pages with missing or malformed data.</li>
      </ul>
    </section>

    <section class="card">
      <h2>Tech Stack</h2>
      <ul>
        <li>Python</li>
        <li>BeautifulSoup4</li>
        <li>Requests</li>
        <li>Pandas</li>
        <li>Jupyter Notebook</li>
      </ul>
    </section>

    <section class="card">
      <h2>Business Impact</h2>
      <p>By automating the extraction of product details, this scraper reduces the time and cost of manual data collection by up to <strong>80%</strong>. The resulting datasets enable:</p>
      <ul>
        <li><strong>Real-time Competitive Price Monitoring:</strong> Identify market shifts and adjust pricing strategies instantly.</li>
        <li><strong>Market Trend Analysis:</strong> Track popular product categories and emerging styles for inventory planning.</li>
        <li><strong>Catalog Building:</strong> Rapidly compile product information for new e-commerce sites or marketplaces.</li>
        <li><strong>Data-Driven Decisions:</strong> Supply chain and marketing teams can leverage clean, structured data for targeted campaigns and demand forecasting.</li>
      </ul>
    </section>

    <section class="card">
      <h2>Potential Applications</h2>
      <ul>
        <li>Competitive intelligence</li>
        <li>Dynamic pricing models</li>
        <li>Product recommendation engines</li>
        <li>Inventory synchronization</li>
      </ul>
    </section>

    <footer>
      <div><strong>Author:</strong> Seun — Supply Chain & Data Enthusiast</div>
      <div><strong>License:</strong> MIT — Free to use with attribution</div>
    </footer>
  </div>
</body>
</html>


`,
    images: ['/images/ecommerce.png'],
    tags: ['WebScraping', 'BeautifulSoup', 'Selenium', 'Scrapy'],
    link: '/project/sentiment-analysis-reviews',
    aiHint: 'cinema film',
    sourceCodeLink: 'https://colab.research.google.com/drive/1cEwXjgRlLRQEO9uv_vCBR9skUF0ma8Q9?usp=drive_open#scrollTo=zkhcGTrMtoJP',
    sentimentAnalysisSection: {
      title: '',
      content: ``

    },
    code: `

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

image_path=os.path.join('.','images','all_image')
os.makedirs(image_path,exist_ok=True)

def savefig(fig_id,tight_layout=True,resolution=300,fig_extension='png'):
    path=os.path.join(image_path,fig_id+'.'+fig_extension)
    print(f'saving figure {fig_id}')
    if tight_layout:
        plt.tight_layout()

    plt.savefig(path,format=fig_extension,dpi=resolution)

from IPython.display import display
sns.set()

# Load Data/webscraping

Route=[]
for i in range(1,100):
    url=f"https://www.airlinequality.com/airline-reviews/british-airways/page/{i}/"
    doc=requests.get(url).text
    result=BeautifulSoup(doc,'lxml')
    for i in range(0,len(result.find_all('td',class_='review-value'))):
        rv=result.find_all('td',class_='review-value')[i].string
        if (re.findall('to',rv)==['to']):
            Route1=(result.find_all('td',class_='review-value')[i].string)
            Route.append(Route1)


rating=[]

for i in range(1,100):
    url=f"https://www.airlinequality.com/airline-reviews/british-airways/page/{i}/"
    doc=requests.get(url).text
    result=BeautifulSoup(doc,'lxml')
    for i in range(0,len(result.div.find_all('time',itemprop="datePublished"))):
            date=result.div.find_all('time',itemprop="datePublished")[i].string
            name=result.div.find_all('span',itemprop="name")[i].string
            country=result.div.find_all('h3')[i].get_text(strip=True)
            comment=result.div.find_all('h2')[i+1].string

            rv=result.find_all('td',class_='review-value')[i].string
            if (re.findall('Leisure$',rv)==['Leisure'])|(re.findall('Business',rv)==['Business']):
                Type_Of_Traveller=(result.find_all('td',class_='review-value')[i].string)

            rating.append([date,name,country,comment,Type_Of_Traveller])


# Data Preprocessing

ba1=British_Airway['country'].str.split(')',n=1,expand=True)
British_Airway['country']=ba1[0].str.split('(',n=1,expand=True)[1]
British_Airway.head()

# Data Visualization
# Cabin Staff Service rating
display(rating_by_Seat_Type(groupby_Seat_Type['Cabin_Staff_Service_rating'],'Cabin_Staff_Service_rating_by_Seat_Type','Cabin Staff Service rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Food_and_Beverages_rating'],'Food_and_Beverages_rating_by_Seat_Type','Food and Beverages rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Ground_Service_rating'],'Ground_Service_rating_by_Seat_Type','Ground Service rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Inflight_Entertainment'],'Inflight_Entertainment_by_Seat_Type','Inflight Entertainment'))
display(rating_by_Seat_Type(groupby_Seat_Type['Seat_comfort_rating'],'Seat_comfort_rating_by_Seat_Type','Seat comfort rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Value_For_Money_rating'],'Value_For_Money_rating_by_Seat_Type','Value For Money rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Wifi_and_Connectivity_rating'],'Wifi_and_Connectivity_rating_by_Seat_Type','Wifi and Connectivity rating'))
 
` 
  },

  {
    id: 16,
    slug: 'jumia-website-scraping',
    title: 'Jumia Website Scraping',
    description: "This project uses requests and BeautifulSoup to scrape product listings and prices from Jumia Nigeria. The scraper works across multiple pages and categories, returning structured datasets ready for analysis.",
    longDescription: `
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Jumia Price & Item Web Scraper</title>
  <meta name="description" content="Python web scraper for extracting product names and prices from Jumia, enabling market analysis, competitive pricing, and inventory planning." />
  <style>
    :root {
      --bg: #ffffff;
      --card: #f7f9fb;
      --muted: #6b7280;
      --accent: #ff6a00;
      --mono: 'Courier New', monospace;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    body {
      background: var(--bg);
      color: #0f172a;
      margin: 0;
      padding: 40px;
      line-height: 1.5;
    }
    .container {
      max-width: 900px;
      margin: 0 auto;
    }
    header {
      display: flex;
      align-items: center;
      gap: 18px;
      margin-bottom: 18px;
    }
    .logo {
      width: 64px;
      height: 64px;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--accent), #ffb366);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #fff;
      font-weight: 700;
      font-family: var(--mono);
    }
    h1 {
      margin: 0;
      font-size: 26px;
    }
    p.lead {
      margin: 8px 0 20px;
      color: var(--muted);
    }
    .card {
      background: var(--card);
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 14px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.05);
    }
    h2 {
      font-size: 18px;
      margin: 0 0 8px;
    }
    ul {
      margin: 0 0 12px 20px;
    }
    .meta {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 14px;
    }
    .badge {
      display: inline-block;
      background: #fff3e6;
      color: var(--accent);
      padding: 6px 8px;
      border-radius: 999px;
      font-weight: 600;
      font-size: 12px;
    }
    code {
      background: #fff0e6;
      padding: 4px 6px;
      border-radius: 6px;
      font-family: var(--mono);
      font-size: 13px;
    }
    footer {
      margin-top: 26px;
      color: var(--muted);
      font-size: 13px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">JU</div>
      <div>
        <h1>Jumia Price & Item Web Scraper</h1>
        <p class="lead">Automated Python scraper that extracts product names and prices from Jumia, enabling competitive price analysis and inventory planning.</p>
        <div class="meta">
          <span class="badge">Portfolio Project</span>
          <span>BeautifulSoup • Requests • Pandas</span>
          <span>Data Collection • Market Analysis</span>
        </div>
      </div>
    </header>

    <section class="card">
      <h2>Overview</h2>
      <p>This project uses <code>requests</code> and <code>BeautifulSoup</code> to scrape product listings and prices from <strong>Jumia Nigeria</strong>. The scraper works across multiple pages and categories, returning structured datasets ready for analysis.</p>
    </section>

    <section class="card">
      <h2>Key Features</h2>
      <ul>
        <li><strong>Category-based Scraping:</strong> Easily target different product categories such as Health & Beauty, Electronics, and more.</li>
        <li><strong>Multi-Page Data Extraction:</strong> Loops through 50+ pages per category to collect extensive datasets.</li>
        <li><strong>Structured Output:</strong> Returns product names and prices in a Pandas DataFrame, exportable to CSV.</li>
        <li><strong>Flexible Function:</strong> <code>data_scraping(category, type)</code> enables quick adaptation to category-specific HTML structures.</li>
      </ul>
    </section>

    <section class="card">
      <h2>Tech Stack</h2>
      <ul>
        <li>Python</li>
        <li>BeautifulSoup4</li>
        <li>Requests</li>
        <li>Pandas</li>
        <li>Jupyter Notebook</li>
      </ul>
    </section>

    <section class="card">
      <h2>Business Impact</h2>
      <p>This scraper provides a rapid, automated way to collect real-time market data from Jumia, delivering measurable business benefits:</p>
      <ul>
        <li><strong>Competitive Pricing Intelligence:</strong> Enables businesses to adjust pricing strategies dynamically to remain competitive.</li>
        <li><strong>Market Trend Monitoring:</strong> Tracks shifts in popular products and pricing over time.</li>
        <li><strong>Inventory Planning:</strong> Helps procurement teams make informed stock replenishment decisions based on market demand and price movement.</li>
        <li><strong>Operational Efficiency:</strong> Reduces manual research time by over <strong>80%</strong>, allowing teams to focus on analysis rather than collection.</li>
      </ul>
    </section>

    <footer>
      <div><strong>Author:</strong> Seun — Supply Chain & Data Enthusiast</div>
      <div><strong>License:</strong> MIT — Free to use with attribution</div>
    </footer>
  </div>
</body>
</html>


`,
    images: ['/images/jumia2.png'],
    tags: ['WebScraping', 'BeautifulSoup', 'Selenium', 'Scrapy'],
    link: '/project/sentiment-analysis-reviews',
    aiHint: 'cinema film',
    sourceCodeLink: 'https://github.com/sheunq/Jumia-/tree/main',
    sentimentAnalysisSection: {
      title: ``,
      content: ``

    },
    code: `

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

image_path=os.path.join('.','images','all_image')
os.makedirs(image_path,exist_ok=True)

def savefig(fig_id,tight_layout=True,resolution=300,fig_extension='png'):
    path=os.path.join(image_path,fig_id+'.'+fig_extension)
    print(f'saving figure {fig_id}')
    if tight_layout:
        plt.tight_layout()

    plt.savefig(path,format=fig_extension,dpi=resolution)

from IPython.display import display
sns.set()

# Load Data/webscraping

Route=[]
for i in range(1,100):
    url=f"https://www.airlinequality.com/airline-reviews/british-airways/page/{i}/"
    doc=requests.get(url).text
    result=BeautifulSoup(doc,'lxml')
    for i in range(0,len(result.find_all('td',class_='review-value'))):
        rv=result.find_all('td',class_='review-value')[i].string
        if (re.findall('to',rv)==['to']):
            Route1=(result.find_all('td',class_='review-value')[i].string)
            Route.append(Route1)


rating=[]

for i in range(1,100):
    url=f"https://www.airlinequality.com/airline-reviews/british-airways/page/{i}/"
    doc=requests.get(url).text
    result=BeautifulSoup(doc,'lxml')
    for i in range(0,len(result.div.find_all('time',itemprop="datePublished"))):
            date=result.div.find_all('time',itemprop="datePublished")[i].string
            name=result.div.find_all('span',itemprop="name")[i].string
            country=result.div.find_all('h3')[i].get_text(strip=True)
            comment=result.div.find_all('h2')[i+1].string

            rv=result.find_all('td',class_='review-value')[i].string
            if (re.findall('Leisure$',rv)==['Leisure'])|(re.findall('Business',rv)==['Business']):
                Type_Of_Traveller=(result.find_all('td',class_='review-value')[i].string)

            rating.append([date,name,country,comment,Type_Of_Traveller])


# Data Preprocessing

ba1=British_Airway['country'].str.split(')',n=1,expand=True)
British_Airway['country']=ba1[0].str.split('(',n=1,expand=True)[1]
British_Airway.head()

# Data Visualization
# Cabin Staff Service rating
display(rating_by_Seat_Type(groupby_Seat_Type['Cabin_Staff_Service_rating'],'Cabin_Staff_Service_rating_by_Seat_Type','Cabin Staff Service rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Food_and_Beverages_rating'],'Food_and_Beverages_rating_by_Seat_Type','Food and Beverages rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Ground_Service_rating'],'Ground_Service_rating_by_Seat_Type','Ground Service rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Inflight_Entertainment'],'Inflight_Entertainment_by_Seat_Type','Inflight Entertainment'))
display(rating_by_Seat_Type(groupby_Seat_Type['Seat_comfort_rating'],'Seat_comfort_rating_by_Seat_Type','Seat comfort rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Value_For_Money_rating'],'Value_For_Money_rating_by_Seat_Type','Value For Money rating'))
display(rating_by_Seat_Type(groupby_Seat_Type['Wifi_and_Connectivity_rating'],'Wifi_and_Connectivity_rating_by_Seat_Type','Wifi and Connectivity rating'))
 
` 
  },

  {
    id: 17,
    slug: 'Stack-Overflow-Text-Classification',
    title: 'Stack Overflow Text Classification',
    description: "This project involves building a multi-class classification model to predict the tag of a Stack Overflow question based on its content. The goal is to classify programming questions into specific tags such as Python, CSharp, JavaScript, and Java using Natural Language Processing (NLP) techniques.",
    longDescription: `<section style="font-family: Arial, sans-serif; margin: 20px;">
    <h2 style="color: #2c3e50;">Multi-Class Classification on Stack Overflow Questions</h2>
    <p>
        This project involves building a multi-class classification model to predict the tag of a Stack Overflow question based on its content. 
        The goal is to classify programming questions into specific tags such as <strong>Python</strong>, <strong>CSharp</strong>, 
        <strong>JavaScript</strong>, and <strong>Java</strong> using Natural Language Processing (NLP) techniques.
    </p>
    
    <h3 style="color: #34495e;">Key Features:</h3>
    <ul>
        <li><strong>Text Classification:</strong> Uses NLP to analyze programming questions and automatically categorize them into one of four tags.</li>
        <li><strong>Multi-Class Classification:</strong> Handles tasks where each question belongs to one of several possible classes.</li>
        <li><strong>Data Preprocessing:</strong> Cleaned and tokenized thousands of questions, normalizing and transforming text for model training.</li>
        <li><strong>Model Architecture:</strong> Built with TensorFlow, optimized for text input, trained on labeled question data.</li>
        <li><strong>Evaluation and Prediction:</strong> Measured performance using accuracy and predicts the correct tag for unseen questions.</li>
    </ul>
    
    <h3 style="color: #34495e;">Technologies Used:</h3>
    <ul>
        <li>Python – Core programming language for data handling and model building</li>
        <li>TensorFlow/Keras – For deep learning model development</li>
        <li>Pandas & NumPy – For data manipulation</li>
        <li>Matplotlib – For visualizations and performance tracking</li>
    </ul>
    
    <h3 style="color: #34495e;">Business Impact:</h3>
    <ul>
        <li><strong>Enhanced User Experience:</strong> Helps users find answers faster by automatically tagging questions.</li>
        <li><strong>Improved Search Functionality:</strong> Better categorization improves filtering and search results.</li>
        <li><strong>Content Management:</strong> Reduces manual tagging workload, enabling scalable growth of user-generated content.</li>
    </ul>
</section>
`,
    images: ['/images/stack overflow.png'],
    tags: ['Data-Science','Deep-Learning', 'Python',"Numpy","Tensorflow","Keras","Pytorch"],
    link: '/project/customer-churn-prediction',
    sourceCodeLink: 'https://colab.research.google.com/drive/10VZoKZ5TjDe7k17zSxsER5f2sIhCCFLn?usp=drive_open#scrollTo=xC1AqieDkUBr',
    aiHint: 'data analytics',
    sentimentAnalysisSection: {
// Suggested code may be subject to a license. Learn more: ~LicenseLog:1192901264.
      title: ``,
      content: ``
    },
    code: `
import tensorflow as tf
import os
import re
import shutil
import pandas as pd
import numpy as np
import string
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
url='https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz'
data=tf.keras.utils.get_file('stack_overflow_16k',url,untar=True, cache_dir='.',
                                    cache_subdir='')

# Preprocessing
def standadization(input_data):
  lower_data=tf.strings.lower(input_data)
  replace_data=tf.strings.regex_replace(lower_data,'<br />','')
  return tf.strings.regex_replace(replace_data,'[%s]'% re.escape(string.punctuation),'')


# Model Architecture

model=tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(1000,16))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(4))

model.summary()

# Make predictions
export_model =tf.keras.Sequential([vectorize_layer,model,tf.keras.layers.Activation('softmax')])

export_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)
    `},

    {
      id: 18,
      slug: 'Fruits-Image-Classification',
      title: 'Fruits Image Classification',
      description: "This project leverages image classification techniques to classify fruit images into their respective categories using Convolutional Neural Networks (CNN). The dataset used is the Fruits 360 dataset, which includes a wide variety of fruits and vegetables, and the goal was to build a model capable of predicting the type of fruit based on the image.",
      longDescription: `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Fruit Image Classification — CNN (Fruits 360)</title>
  <meta name="description" content="Convolutional Neural Network (CNN) model using the Fruits 360 dataset to classify fruit images. Includes preprocessing, augmentation, model training, evaluation and business applications." />
  <style>
    :root{
      --bg:#ffffff;
      --card:#f6f8fb;
      --muted:#4b5563;
      --accent:#1f7a4c;
      --accent-2:#2aa173;
      --mono:"Courier New", monospace;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    html,body{height:100%}
    body{
      margin:0;
      background:var(--bg);
      color:#0f1724;
      padding:40px;
      line-height:1.55;
      -webkit-font-smoothing:antialiased;
    }
    .container{max-width:980px;margin:0 auto;}
    header{display:flex;align-items:center;gap:16px;margin-bottom:18px}
    .logo{width:64px;height:64px;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent-2));display:flex;align-items:center;justify-content:center;color:#fff;font-weight:800;font-family:var(--mono);font-size:18px}
    h1{margin:0;font-size:24px}
    p.lead{margin:6px 0 14px;color:var(--muted)}
    .card{background:var(--card);border-radius:12px;padding:18px;margin-bottom:14px;box-shadow:0 8px 24px rgba(31,122,76,0.06)}
    h2{font-size:16px;margin:0 0 8px}
    ul{margin:8px 0 12px 20px}
    code{background:#eef9f4;padding:4px 8px;border-radius:6px;font-family:var(--mono);font-size:13px}
    .meta{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px}
    .cta{display:inline-block;margin-top:12px;padding:10px 14px;border-radius:8px;background:var(--accent);color:#fff;text-decoration:none;font-weight:700}
    .two-col{display:grid;grid-template-columns:1fr 320px;gap:16px}
    .sidebar{font-size:14px;color:var(--muted)}
    footer{margin-top:22px;color:var(--muted);font-size:13px}
    .badge{display:inline-block;background:#e8fbef;color:var(--accent-2);padding:6px 8px;border-radius:999px;font-weight:700;font-size:12px}
    @media (max-width:920px){.two-col{grid-template-columns:1fr}.logo{width:56px;height:56px}}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">CV</div>
      <div>
        <h1>Fruit Image Classification — CNN (Fruits 360)</h1>
        <p class="lead">Deep learning pipeline that classifies fruit images using a Convolutional Neural Network trained on the Fruits 360 dataset — production-aware and business-focused.</p>
        <div class="meta">
          <span class="badge">Portfolio Project</span>
          <span>TensorFlow / Keras • Python • CNN</span>
          <span>Image Preprocessing • Augmentation • Evaluation</span>
        </div>
      </div>
    </header>

    <section class="card two-col" aria-labelledby="overview">
      <div>
        <h2 id="overview">Project Overview</h2>
        <p>This project builds a Convolutional Neural Network (CNN) to classify fruit images into their correct categories using the Fruits 360 dataset (apples, bananas, oranges, grapes, and many more). The pipeline emphasizes robust preprocessing, augmentation, and regularization to maximize accuracy and generalization.</p>

        <h2>Key Features</h2>
        <ul>
          <li><strong>Dataset:</strong> Fruits 360 — well-labeled images across multiple fruit classes and varieties.</li>
          <li><strong>Preprocessing:</strong> resize, normalize, and standardize image inputs for model consistency.</li>
          <li><strong>Augmentation:</strong> on-the-fly image augmentation (rotation, flip, zoom, shift) to increase effective dataset size and reduce overfitting.</li>
          <li><strong>CNN Architecture:</strong> layered convolution, pooling, batch normalization, dropout, and dense classification head designed for visual feature extraction.</li>
          <li><strong>Training & Evaluation:</strong> train/validation/test split, use of callbacks (EarlyStopping, ModelCheckpoint), and evaluation via accuracy, precision, recall, and confusion matrices.</li>
          <li><strong>Visualization:</strong> training curves, confusion matrix, and per-class performance plots to guide model diagnostics.</li>
        </ul>

        <h2>Technologies Used</h2>
        <ul>
          <li>Python — orchestration and preprocessing</li>
          <li>TensorFlow & Keras — model definition, training, and deployment artifacts</li>
          <li>NumPy & Pandas — dataset handling</li>
          <li>Matplotlib & Seaborn — performance visualization</li>
          <li>Kaggle — dataset source and experiment tracking</li>
        </ul>

        <h2>How to run</h2>
        <ol>
          <li>Download the Fruits 360 dataset (or point the notebook to the Kaggle dataset).</li>
          <li>Install dependencies: <code>pip install -r requirements.txt</code> (includes tensorflow, numpy, pandas, matplotlib).</li>
          <li>Run preprocessing → create data generators with augmentation → define CNN → train with callbacks.</li>
          <li>Export model: <code>model.save('fruit_cnn.h5')</code> and generate evaluation plots for reporting.</li>
        </ol>
      </div>

      <aside class="sidebar card" aria-labelledby="business">
        <h2 id="business">Business Impact</h2>
        <p>This model delivers operational and commercial value across multiple industries by automating visual identification tasks that are time-consuming and error-prone when done manually.</p>
        <ul>
          <li><strong>E-commerce:</strong> Auto-classify fresh produce images for cataloging, search relevance, and recommendation systems — reducing manual tagging costs and improving UX.</li>
          <li><strong>Agriculture & Supply Chain:</strong> Automate sorting and quality inspection during packing/harvesting to reduce waste and improve throughput.</li>
          <li><strong>Food Processing:</strong> Integrate into conveyor vision systems for real-time categorization and reject faulty items, improving quality control.</li>
          <li><strong>Scalability:</strong> Deployable as an inference service (TensorFlow Serving / REST API) for near real-time predictions in production environments.</li>
        </ul>

        <h2>KPIs & Expected Outcomes</h2>
        <ul>
          <li>Labeling Cost Reduction (↓ manual hours)</li>
          <li>Throughput / Sorting Speed (↑)</li>
          <li>Defect Detection Accuracy (↑)</li>
          <li>Customer Catalog Match Rate (↑) — better search & conversion</li>
        </ul>

        <a class="cta" href="https://colab.research.google.com/drive/10VZoKZ5TjDe7k17zSxsER5f2sIhCCFLn?usp=drive_open#scrollTo=xC1AqieDkUBr" title="Link to notebook or repo">View Notebook / Repo</a>
      </aside>
    </section>

    

    <footer>
      <div><strong>Author:</strong> Seun — Supply Chain & AI Practitioner</div>
      <div><strong>License:</strong> MIT — reuse with attribution.</div>
    </footer>
  </div>
</body>
</html>
`,
      images: ['/images/fruits.png'],
      tags: ['Data-Science','Deep-Learning', 'Python',"Numpy","Tensorflow","Keras","Pytorch"],
      link: '/project/customer-churn-prediction',
      sourceCodeLink: 'https://colab.research.google.com/drive/10VZoKZ5TjDe7k17zSxsER5f2sIhCCFLn?usp=drive_open#scrollTo=xC1AqieDkUBr',
      aiHint: 'data analytics',
      sentimentAnalysisSection: {
  // Suggested code may be subject to a license. Learn more: ~LicenseLog:1192901264.
        title: 'Project Overview',
        content: "Customers who left within the last month - the column is called Churn Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies Customer account information - how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges Demographic info about customers - gender, age range, and if they have partners and dependents Inspiration To explore this type of models and learn more about the subject."
      },
      code: `
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
import datetime
  
# Load the dataset

train_ds=(os.path.join('/content/fruits','fruits-360-original-size','fruits-360-original-size','Training'))
test_ds=(os.path.join('/content/fruits','fruits-360-original-size','fruits-360-original-size','Test'))
val_ds=(os.path.join('/content/fruits','fruits-360-original-size','fruits-360-original-size','Validation'))
  
# Preprocessing

autotune=tf.data.AUTOTUNE
train_ds=train_ds.cache().prefetch(buffer_size=autotune)
test_ds=test_ds.cache().prefetch(buffer_size=autotune)
val_ds=val_ds.cache().prefetch(buffer_size=autotune)
  
# Model Architecture
  

  model=tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(30,4,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(60,5,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(90,6,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(class_num))

    #tf.keras.layers.Softmax()
])

# Make predictions
  checkpoint_path='training_1/cp.ckpt'
checkpoint_dir=os.path.dirname(checkpoint_path)
cp_calllback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
  )
      `}
      ,
      {
        id: 19,
        slug: 'Purchase-Prediction',
        title: 'Purchase Prediction',
        description: "Purchase Prediction with Logistic Regression.\
This project focuses on using logistic regression to predict customer purchases based on various demographic and behavioral features. The goal was to predict whether a customer will make a purchase (binary classification) using factors such as age, gender, marital status, income, occupation, and more.",
        longDescription: `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Purchase Prediction with Logistic Regression</title>
  <meta name="description" content="Logistic regression model to predict customer purchases using demographic and behavioral features. Includes data preprocessing, feature selection, model training, and visualizations." />
  <style>
    :root{
      --bg: #ffffff;
      --card: #f6f8fb;
      --muted: #525866;
      --accent: #0c6b8f;
      --accent-2: #0b9bdc;
      --mono: "Courier New", monospace;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    html,body{height:100%}
    body{
      margin:0;
      background:var(--bg);
      color:#0f1724;
      padding:40px;
      line-height:1.5;
      -webkit-font-smoothing:antialiased;
    }
    .container{max-width:920px;margin:0 auto;}
    header{display:flex;align-items:center;gap:16px;margin-bottom:18px}
    .logo{width:64px;height:64px;border-radius:12px;background:linear-gradient(135deg,var(--accent),var(--accent-2));display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;font-family:var(--mono);font-size:18px}
    h1{margin:0;font-size:24px}
    p.lead{margin:6px 0 14px;color:var(--muted)}
    .card{background:var(--card);border-radius:12px;padding:18px;margin-bottom:14px;box-shadow:0 8px 24px rgba(11,107,143,0.06)}
    h2{font-size:16px;margin:0 0 8px}
    ul{margin:8px 0 12px 20px}
    code{background:#eaf7fb;padding:5px 8px;border-radius:6px;font-family:var(--mono);font-size:13px}
    .meta{display:flex;gap:12px;flex-wrap:wrap;color:var(--muted);font-size:13px}
    .cta{display:inline-block;margin-top:10px;padding:9px 12px;border-radius:8px;background:var(--accent);color:#fff;text-decoration:none;font-weight:600}
    .two-col{display:grid;grid-template-columns:1fr 300px;gap:16px}
    .sidebar{font-size:14px;color:var(--muted)}
    footer{margin-top:22px;color:var(--muted);font-size:13px}
    .badge{display:inline-block;background:#e6f6fb;color:var(--accent-2);padding:6px 8px;border-radius:999px;font-weight:700;font-size:12px}
    @media (max-width:860px){.two-col{grid-template-columns:1fr}.logo{width:56px;height:56px}}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo">ML</div>
      <div>
        <h1>Purchase Prediction with Logistic Regression</h1>
        <p class="lead">Binary classification pipeline using logistic regression to predict customer purchases from demographic and behavioral features — practical, production-minded, and business-driven.</p>
        <div class="meta">
          <span class="badge">Portfolio Project</span>
          <span>Python • scikit-learn • pandas</span>
          <span>EDA • Modeling • Visualization</span>
        </div>
      </div>
    </header>

    <section class="card two-col" aria-labelledby="overview">
      <div>
        <h2 id="overview">Project Overview</h2>
        <p>This project builds a logistic regression model to predict whether a customer will make a purchase (binary outcome) using features such as <strong>age, gender, marital status, income, occupation</strong>, and other behavioral attributes. The focus is on rigorous data preparation, informed feature selection, model training, evaluation, and visualization to drive business decisions.</p>

        <h2>Key Features</h2>
        <ul>
          <li><strong>Data Preprocessing:</strong> cleaning, handling missing values, encoding categorical variables, and scaling as appropriate for logistic regression.</li>
          <li><strong>Feature Selection:</strong> identification of high-signal features (age, marital status, income, occupation) to reduce noise and improve model interpretability.</li>
          <li><strong>Logistic Regression Model:</strong> training and evaluation of a logistic regression classifier with standard metrics (accuracy, precision, recall, F1, ROC-AUC).</li>
          <li><strong>Evaluation & Validation:</strong> K-fold cross-validation and a hold-out test set to measure generalization performance and avoid overfitting.</li>
          <li><strong>Visualization:</strong> exploratory visualizations and model performance plots (ROC curve, confusion matrix) for stakeholder reporting.</li>
        </ul>

        <h2>Technologies Used</h2>
        <ul>
          <li>Python — orchestration and preprocessing</li>
          <li>scikit-learn — logistic regression, metrics, model selection</li>
          <li>pandas — data wrangling</li>
          <li>Matplotlib & Seaborn — EDA and model visualization</li>
        </ul>

        <h2>How to run</h2>
        <ol>
          <li>Clone the repo and open the notebook/script in Jupyter or run the pipeline as a Python script.</li>
          <li>Install dependencies: <code>pip install -r requirements.txt</code> (includes scikit-learn, pandas, matplotlib, seaborn).</li>
          <li>Run preprocessing → feature selection → model training cells in order. Export model artifacts and visualizations as needed.</li>
        </ol>
      </div>

      <aside class="sidebar card" aria-labelledby="business">
        <h2 id="business">Business Impact</h2>
        <p>This model delivers tangible commercial value by enabling teams to prioritize high-propensity customers and optimize resource allocation. Key business outcomes include:</p>
        <ul>
          <li><strong>Targeted Marketing:</strong> Increase campaign ROI by focusing offers on customers with higher predicted purchase probabilities.</li>
          <li><strong>Resource Allocation:</strong> Reduce marketing spend waste and reassign budget to high-conversion segments — improving conversion efficiency.</li>
          <li><strong>Customer Insights:</strong> Understand demographic and behavioral drivers of purchase behavior to inform promotions, product placement, and assortment strategies.</li>
          <li><strong>Operational Efficiency:</strong> Shorten decision cycles for sales and retention teams by providing predictive signals to action in near real-time.</li>
        </ul>

        <h2>Typical KPIs improved</h2>
        <ul>
          <li>Conversion Rate (↑)</li>
          <li>Marketing Cost per Acquisition (↓)</li>
          <li>Customer Lifetime Value (↑)</li>
          <li>Campaign ROI (↑)</li>
        </ul>

        <a class="cta" href="https://www.kaggle.com/code/seunayegboyin/purchase-prediction-with-logistic-regression#Linear-Regression" title="Link to notebook or repo">View Notebook / Repo</a>
      </aside>
    
    <footer>
      <div><strong>Author:</strong> Seun — Supply Chain & Data Practitioner</div>
      <div><strong>License:</strong> MIT — reuse with attribution.</div>
    </footer>
  </div>
</body>
</html>
`,
        images: ['/images/purchase prediction.png'],
        tags: ['Data-Science','Scikit-learn', 'Python',"Numpy",'K-Means',"PCA",'Logistic Regression'],
        link: '/project/customer-churn-prediction',
        sourceCodeLink: 'https://www.kaggle.com/code/seunayegboyin/purchase-prediction-with-logistic-regression#Linear-Regression',
        aiHint: 'data analytics',
        sentimentAnalysisSection: {
    // Suggested code may be subject to a license. Learn more: ~LicenseLog:1192901264.
          title: 'Project Overview',
          content: "Customers who left within the last month - the column is called Churn Services that each customer has signed up for - phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies Customer account information - how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges Demographic info about customers - gender, age range, and if they have partners and dependents Inspiration To explore this type of models and learn more about the subject."
        },
        code: `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

sns.set()
    
  # Load the dataset
  
 purch=pd.read_csv('../input/purchase/purchase data.csv')
purch.sample(6)
    
  # Preprocessing
  
purch_analy=purch[['ID','Incidence']].groupby('ID').count().rename(columns={"Incidence":"Number of visit"})
purch_incidence=purch[purch["Incidence"]==1].groupby("ID").sum()
purch_analy["Number of Purchase"]=purch_incidence["Incidence"]
purch_analy["Average Purchase"]=(purch_analy["Number of Purchase"]/purch_analy["Number of visit"]).round(3)
purch_analy.head().sort_values(['Number of visit','Number of Purchase','Average Purchase'])
purch_seg=purch[['ID',"segment"]].groupby('ID').mean().round()
purch_analy['segment']=purch_seg['segment']
purch_analy
    
  # Model
    
  purc_quant=purch[purch["Incidence"]==1]
purc_quant=pd.get_dummies(purc_quant,columns=['Brand'],prefix='Brand',prefix_sep='_')
purc_quant['Price_Incidence']=(purc_quant['Brand_1']*purc_quant["Price_1"]+
                              purc_quant['Brand_2']*purc_quant["Price_2"]+
                              purc_quant['Brand_3']*purc_quant["Price_3"]+
                              purc_quant['Brand_4']*purc_quant["Price_4"]+
                              purc_quant['Brand_5']*purc_quant["Price_5"])
purc_quant['Promotion_Incidence']=(purc_quant['Brand_1']*purc_quant["Promotion_1"]+
                              purc_quant['Brand_2']*purc_quant["Promotion_2"]+
                              purc_quant['Brand_3']*purc_quant["Promotion_3"]+
                              purc_quant['Brand_4']*purc_quant["Promotion_4"]+
                              purc_quant['Brand_5']*purc_quant["Promotion_5"])
  
  
  # Make predictions
  x= purc_quant[['Price_Incidence','Promotion_Incidence']]
y=purc_quant["Quantity"]
model_quantity=LinearRegression()
model_quantity.fit(x,y)
model_quantity.coef_
        `}
];

const blogPosts: BlogPost[] = [
  {
    id: 1,
    slug: 'understanding-auc-roc-curve',
    title: 'Understanding AUC-ROC Curve',
    summary: 'A comprehensive guide to understanding the AUC-ROC curve, a key metric for evaluating classification models in machine learning.',
    content: '<p>The AUC-ROC curve is a fundamental tool for any data scientist. ROC stands for Receiver Operating Characteristic, and AUC is the Area Under the Curve. It provides an aggregate measure of performance across all possible classification thresholds.</p><p>A model with a higher AUC is better at distinguishing between positive and negative classes. An AUC of 1.0 represents a perfect model, while an AUC of 0.5 suggests a model with no discriminative power, equivalent to random guessing.</p>',
    author: 'Seun Ayegboyin',
    date: '2024-10-26',
    image: '/images/roc.png',
    aiHint: 'data visualization',
  },
  {
    id: 2,
    slug: 'getting-started-with-pandas',
    title: 'Getting Started with Pandas for Data Analysis',
    summary: 'Pandas is the most popular Python library for data manipulation and analysis. This post covers the basics to get you started.',
    content: '<p>Pandas is built on top of NumPy and provides easy-to-use data structures and data analysis tools. The primary data structures in Pandas are the `Series` (1-dimensional) and the `DataFrame` (2-dimensional).</p><p>This tutorial walks you through reading data from a CSV file, inspecting the DataFrame, selecting columns, filtering rows, and handling missing values. It\'s the first step towards mastering data wrangling in Python.</p>',
    author: 'Seun Ayegboyin',
    date: '2024-11-15',
    image: '/images/data analysis.png',
    aiHint: 'python code',
  },
  {
    id: 3,
    slug: 'introduction-to-natural-language-processing',
    title: 'An Introduction to Natural Language Processing',
    summary: 'Explore the fascinating world of Natural Language Processing (NLP) and how computers can be taught to understand human language.',
    content: '<p>Natural Language Processing (NLP) is a subfield of artificial intelligence that is focused on enabling computers to understand, interpret, and manipulate human language. NLP has a wide range of applications, from spam detection and machine translation to sentiment analysis and chatbots.</p><p>This article covers core NLP concepts like tokenization, stemming, lemmatization, and part-of-speech-tagging. It also provides a high-level overview of popular NLP libraries like NLTK and spaCy.</p>',
    author: 'Seun Ayegboyin',
    date: '2024-12-05',
    image: '/images/NLP.png',
    aiHint: 'robot thinking',
  },
];

export function getProjectCategories(): ProjectCategory[] {
  return projectCategories;
}

export function getProjects(tag?: string): Project[] {
  if (tag) {
    return projects.filter(p => p.tags.includes(tag));
  }
  return projects;
}

export function getProjectBySlug(slug: string): Project | undefined {
  return projects.find((project) => project.slug === slug);
}

export function getBlogPosts(): BlogPost[] {
  return blogPosts;
}

export function getBlogPostBySlug(slug: string): BlogPost | undefined {
  return blogPosts.find((post) => post.slug === slug);
}

export function getProjectTags(): string[] {
  const tags = new Set<string>();
  projects.forEach(p => {
    p.tags.forEach(t => tags.add(t));
  });
  return Array.from(tags);
}
