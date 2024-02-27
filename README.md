# Python-ML

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f345a6f4",
   "metadata": {},
   "source": [
    "# Logistical Regression Analysis:\n",
    "\n",
    "### Covers data loading, preprocessing, exploratory data analysis, feature selection, model training, evaluation, result interpretation and prediction for a new input.\n",
    "\n",
    "Introduction This project aims to predict whether it will rain tomorrow in Australia based on historical weather data using a logistic regression model.\n",
    "\n",
    "Dataset context This dataset sourced from kaggle contains about 10 years of daily weather observations from numerous Australian weather stations.\n",
    "\n",
    "RainTomorrow is the target variable to predict. It means, did it rain the next day - Yes or No? This column is Yes if the rain for that day was 1mm or more.b"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "73df0238",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries for data analysis and modeling\n",
    "\n",
    "import pandas as pd # For data manipulation and analysis\n",
    "import numpy as np # For numerical computations\n",
    "import matplotlib.pyplot as plt # For data visualisation\n",
    "import seaborn as sns # For enhanced data visualisation\n",
    "\n",
    "from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets\n",
    "from sklearn.preprocessing import StandardScaler  # For feature standardisation\n",
    "from sklearn.impute import SimpleImputer  # For handling missing values\n",
    "from sklearn.linear_model import LogisticRegression  # For logistic regression model\n",
    "from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  # For model evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4cb900e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Date</th>\n",
       "      <th>Location</th>\n",
       "      <th>MinTemp</th>\n",
       "      <th>MaxTemp</th>\n",
       "      <th>Rainfall</th>\n",
       "      <th>Evaporation</th>\n",
       "      <th>Sunshine</th>\n",
       "      <th>WindGustDir</th>\n",
       "      <th>WindGustSpeed</th>\n",
       "      <th>WindDir9am</th>\n",
       "      <th>...</th>\n",
       "      <th>Humidity9am</th>\n",
       "      <th>Humidity3pm</th>\n",
       "      <th>Pressure9am</th>\n",
       "      <th>Pressure3pm</th>\n",
       "      <th>Cloud9am</th>\n",
       "      <th>Cloud3pm</th>\n",
       "      <th>Temp9am</th>\n",
       "      <th>Temp3pm</th>\n",
       "      <th>RainToday</th>\n",
       "      <th>RainTomorrow</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2008-12-01</td>\n",
       "      <td>Albury</td>\n",
       "      <td>13.4</td>\n",
       "      <td>22.9</td>\n",
       "      <td>0.6</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>W</td>\n",
       "      <td>44.0</td>\n",
       "      <td>W</td>\n",
       "      <td>...</td>\n",
       "      <td>71.0</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1007.7</td>\n",
       "      <td>1007.1</td>\n",
       "      <td>8.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>16.9</td>\n",
       "      <td>21.8</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2008-12-02</td>\n",
       "      <td>Albury</td>\n",
       "      <td>7.4</td>\n",
       "      <td>25.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>WNW</td>\n",
       "      <td>44.0</td>\n",
       "      <td>NNW</td>\n",
       "      <td>...</td>\n",
       "      <td>44.0</td>\n",
       "      <td>25.0</td>\n",
       "      <td>1010.6</td>\n",
       "      <td>1007.8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>17.2</td>\n",
       "      <td>24.3</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2008-12-03</td>\n",
       "      <td>Albury</td>\n",
       "      <td>12.9</td>\n",
       "      <td>25.7</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>WSW</td>\n",
       "      <td>46.0</td>\n",
       "      <td>W</td>\n",
       "      <td>...</td>\n",
       "      <td>38.0</td>\n",
       "      <td>30.0</td>\n",
       "      <td>1007.6</td>\n",
       "      <td>1008.7</td>\n",
       "      <td>NaN</td>\n",
       "      <td>2.0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>23.2</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2008-12-04</td>\n",
       "      <td>Albury</td>\n",
       "      <td>9.2</td>\n",
       "      <td>28.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NE</td>\n",
       "      <td>24.0</td>\n",
       "      <td>SE</td>\n",
       "      <td>...</td>\n",
       "      <td>45.0</td>\n",
       "      <td>16.0</td>\n",
       "      <td>1017.6</td>\n",
       "      <td>1012.8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>18.1</td>\n",
       "      <td>26.5</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2008-12-05</td>\n",
       "      <td>Albury</td>\n",
       "      <td>17.5</td>\n",
       "      <td>32.3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>W</td>\n",
       "      <td>41.0</td>\n",
       "      <td>ENE</td>\n",
       "      <td>...</td>\n",
       "      <td>82.0</td>\n",
       "      <td>33.0</td>\n",
       "      <td>1010.8</td>\n",
       "      <td>1006.0</td>\n",
       "      <td>7.0</td>\n",
       "      <td>8.0</td>\n",
       "      <td>17.8</td>\n",
       "      <td>29.7</td>\n",
       "      <td>No</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 23 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "         Date Location  MinTemp  MaxTemp  Rainfall  Evaporation  Sunshine  \\\n",
       "0  2008-12-01   Albury     13.4     22.9       0.6          NaN       NaN   \n",
       "1  2008-12-02   Albury      7.4     25.1       0.0          NaN       NaN   \n",
       "2  2008-12-03   Albury     12.9     25.7       0.0          NaN       NaN   \n",
       "3  2008-12-04   Albury      9.2     28.0       0.0          NaN       NaN   \n",
       "4  2008-12-05   Albury     17.5     32.3       1.0          NaN       NaN   \n",
       "\n",
       "  WindGustDir  WindGustSpeed WindDir9am  ... Humidity9am  Humidity3pm  \\\n",
       "0           W           44.0          W  ...        71.0         22.0   \n",
       "1         WNW           44.0        NNW  ...        44.0         25.0   \n",
       "2         WSW           46.0          W  ...        38.0         30.0   \n",
       "3          NE           24.0         SE  ...        45.0         16.0   \n",
       "4           W           41.0        ENE  ...        82.0         33.0   \n",
       "\n",
       "   Pressure9am  Pressure3pm  Cloud9am  Cloud3pm  Temp9am  Temp3pm  RainToday  \\\n",
       "0       1007.7       1007.1       8.0       NaN     16.9     21.8         No   \n",
       "1       1010.6       1007.8       NaN       NaN     17.2     24.3         No   \n",
       "2       1007.6       1008.7       NaN       2.0     21.0     23.2         No   \n",
       "3       1017.6       1012.8       NaN       NaN     18.1     26.5         No   \n",
       "4       1010.8       1006.0       7.0       8.0     17.8     29.7         No   \n",
       "\n",
       "   RainTomorrow  \n",
       "0            No  \n",
       "1            No  \n",
       "2            No  \n",
       "3            No  \n",
       "4            No  \n",
       "\n",
       "[5 rows x 23 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Load the dataset and Head \n",
    "\n",
    "df = pd.read_csv('weatherAUS.csv')\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a05978c8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(145460, 23)"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Shape of Data\n",
    "\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "db639ef9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 145460 entries, 0 to 145459\n",
      "Data columns (total 23 columns):\n",
      " #   Column         Non-Null Count   Dtype  \n",
      "---  ------         --------------   -----  \n",
      " 0   Date           145460 non-null  object \n",
      " 1   Location       145460 non-null  object \n",
      " 2   MinTemp        143975 non-null  float64\n",
      " 3   MaxTemp        144199 non-null  float64\n",
      " 4   Rainfall       142199 non-null  float64\n",
      " 5   Evaporation    82670 non-null   float64\n",
      " 6   Sunshine       75625 non-null   float64\n",
      " 7   WindGustDir    135134 non-null  object \n",
      " 8   WindGustSpeed  135197 non-null  float64\n",
      " 9   WindDir9am     134894 non-null  object \n",
      " 10  WindDir3pm     141232 non-null  object \n",
      " 11  WindSpeed9am   143693 non-null  float64\n",
      " 12  WindSpeed3pm   142398 non-null  float64\n",
      " 13  Humidity9am    142806 non-null  float64\n",
      " 14  Humidity3pm    140953 non-null  float64\n",
      " 15  Pressure9am    130395 non-null  float64\n",
      " 16  Pressure3pm    130432 non-null  float64\n",
      " 17  Cloud9am       89572 non-null   float64\n",
      " 18  Cloud3pm       86102 non-null   float64\n",
      " 19  Temp9am        143693 non-null  float64\n",
      " 20  Temp3pm        141851 non-null  float64\n",
      " 21  RainToday      142199 non-null  object \n",
      " 22  RainTomorrow   142193 non-null  object \n",
      "dtypes: float64(16), object(7)\n",
      "memory usage: 25.5+ MB\n"
     ]
    }
   ],
   "source": [
    "# Info\n",
    "\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9c868e80",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Date             0\n",
      "Location         0\n",
      "MinTemp          0\n",
      "MaxTemp          0\n",
      "Rainfall         0\n",
      "Evaporation      0\n",
      "Sunshine         0\n",
      "WindGustDir      0\n",
      "WindGustSpeed    0\n",
      "WindDir9am       0\n",
      "WindDir3pm       0\n",
      "WindSpeed9am     0\n",
      "WindSpeed3pm     0\n",
      "Humidity9am      0\n",
      "Humidity3pm      0\n",
      "Pressure9am      0\n",
      "Pressure3pm      0\n",
      "Cloud9am         0\n",
      "Cloud3pm         0\n",
      "Temp9am          0\n",
      "Temp3pm          0\n",
      "RainToday        0\n",
      "RainTomorrow     0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Identify columns with missing values\n",
    "columns_with_missing = df.columns[df.isnull().any()]\n",
    "\n",
    "# Handle missing values\n",
    "for column in columns_with_missing:\n",
    "    # Numerical columns: impute with mean\n",
    "    if df[column].dtype == 'float64':\n",
    "        imputer = SimpleImputer(strategy='mean')\n",
    "        df[column] = imputer.fit_transform(df[[column]])\n",
    "    # Categorical columns: impute with mode\n",
    "    else:\n",
    "        imputer = SimpleImputer(strategy='most_frequent')\n",
    "        df[column] = imputer.fit_transform(df[[column]])\n",
    "\n",
    "# Verify that all missing values have been handled\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3cb4adea",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "f1c10581",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "No     113583\n",
       "Yes     31877\n",
       "Name: RainTomorrow, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Out Target Variable is Rain\n",
    "\n",
    "df['RainTomorrow'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "5487b71c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAtEAAAIjCAYAAADFk0cVAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABByElEQVR4nO3de3zP9f//8ft7m22YbY6bZaH4YAw5NHOWZU712SeVSUKij0ZOORVLEkXORD4d6OAT6pN8qbHmGBKTMjnVR1Ha+DT2zmJje/3+6LLXz7sNe87YcLteLu/Lxev5erxfz8frvXfr7uX1fr4dlmVZAgAAAJBvbkXdAAAAAHCjIUQDAAAAhgjRAAAAgCFCNAAAAGCIEA0AAAAYIkQDAAAAhgjRAAAAgCFCNAAAAGCIEA0AAAAYIkQDKFYmTJggh8NxXeZq27at2rZta29v3LhRDodDH3744XWZv0+fPqpWrdp1maugzpw5oyeeeEKBgYFyOBwaOnTodZ3/er4fAMAEIRrANbN48WI5HA774e3traCgIEVGRmrOnDn6/fffC2We48ePa8KECdqzZ0+hHK8wFefe8mPy5MlavHixBg4cqHfffVe9evW6ZG21atVcft6lS5fW3XffrXfeeec6dnxpbdu2denvUo8JEyYUdasAbgAOy7Ksom4CwM1p8eLF6tu3ryZOnKjq1avr/PnzSk5O1saNGxUfH6/bb79dq1atUv369e3nXLhwQRcuXJC3t3e+59m1a5eaNm2qt99+W3369Mn38zIzMyVJnp6ekv68Et2uXTutWLFCDz74YL6PU9Dezp8/r+zsbHl5eRXKXNdCs2bN5OHhoS+++OKKtdWqVVPZsmU1YsQISdKvv/6qN954Q4cOHdKiRYvUv39/4/kL8n64lPj4eKWkpNjbO3fu1Jw5c/Tss8+qTp069nj9+vVd3pMAkBePom4AwM2vU6dOatKkib09duxYrV+/Xl27dtX999+v/fv3q2TJkpIkDw8PeXhc219Nf/zxh0qVKmWH56JSokSJIp0/P06cOKGQkJB8199222169NFH7e0+ffrojjvu0MyZMwsUogvz/XDvvfe6bHt7e2vOnDm69957XW7rKe7S09NVunTpPPflvLcBXHvczgGgSNxzzz0aP368fvrpJ7333nv2eF73wMbHx6tly5by9/eXj4+PatWqpWeffVbSn1ePmzZtKknq27ev/U/yixcvlvTnP+HXq1dPiYmJat26tUqVKmU/96/3ROfIysrSs88+q8DAQJUuXVr333+/jh075lJTrVq1PK96X3zMK/WW1z3R6enpGjFihIKDg+Xl5aVatWrp1Vdf1V//0dDhcGjQoEFauXKl6tWrJy8vL9WtW1dxcXF5v+B/ceLECfXr108BAQHy9vZWgwYNtGTJEnt/zv3hR44c0Zo1a+zef/zxx3wdP0fFihVVu3Zt/fDDDy7jW7Zs0UMPPaTbb79dXl5eCg4O1rBhw3T27FmXurzeD1d77lfy2muvqW7duvLy8lJQUJBiYmJ0+vRpl5qc99W3336rNm3aqFSpUqpRo4Z9P/2mTZsUFhamkiVLqlatWvr8889zzfP111+rU6dO8vX1lY+Pj9q3b68vv/zSpSbnlqhNmzbpqaeeUqVKlVSlShWXHvJ6b1/p5ytJjRo10gMPPOAyFhoaKofDoW+//dYeW7ZsmRwOh/bv31+wFxS4SRGiARSZnPtr161bd8maffv2qWvXrsrIyNDEiRM1ffp03X///dq6daskqU6dOpo4caIkacCAAXr33Xf17rvvqnXr1vYxfvvtN3Xq1EkNGzbUrFmz1K5du8v29dJLL2nNmjUaPXq0nn76acXHxysiIiJXwLuS/PR2McuydP/992vmzJnq2LGjZsyYoVq1amnkyJEaPnx4rvovvvhCTz31lKKjozV16lSdO3dO3bp102+//XbZvs6ePau2bdvq3XffVc+ePTVt2jT5+fmpT58+mj17tt37u+++qwoVKqhhw4Z27xUrVjR6DS5cuKCff/5ZZcuWdRlfsWKF/vjjDw0cOFBz585VZGSk5s6dq8ceeyxfxy3ouV/JhAkTFBMTo6CgIE2fPl3dunXT66+/rg4dOuj8+fMutadOnVLXrl0VFhamqVOnysvLS9HR0Vq2bJmio6PVuXNnvfzyy0pPT9eDDz7o8hmAffv2qVWrVvrmm280atQojR8/XkeOHFHbtm21Y8eOXH099dRT+u677xQbG6sxY8bY43m9t/Pz85WkVq1audymk5qaqn379snNzU1btmyxx7ds2aKKFSu63PICQJIFANfI22+/bUmydu7ceckaPz8/66677rK3n3/+eeviX00zZ860JFknT5685DF27txpSbLefvvtXPvatGljSbIW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u2bz5s06f/68XRMfH69atWqpbNmyds3F8+TU5MwDAACAW0exDtH33XefXnrpJa1Zs0Y//vijPv74Y82YMUP/+Mc/JEkOh0NDhw7VpEmTtGrVKu3du1ePPfaYgoKCFBUVJUmqU6eOOnbsqP79++urr77S1q1bNWjQIEVHRysoKEiS9Mgjj8jT01P9+vXTvn37tGzZMs2ePVvDhw+3exkyZIji4uI0ffp0HThwQBMmTNCuXbs0aNCg6/66AAAAoGgV69s55s6dq/Hjx+upp57SiRMnFBQUpCeffFKxsbF2zahRo5Senq4BAwbo9OnTatmypeLi4uTt7W3XvP/++xo0aJDat28vNzc3devWTXPmzLH3+/n5ad26dYqJiVHjxo1VoUIFxcbGuqwl3bx5cy1dulTjxo3Ts88+q5o1a2rlypWqV6/e9XkxAAAAUGwU63WibzasEw3gWmGdaAAoHDfFOtEAAABAcUSIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwVKETfcccd+u2333KNnz59WnfcccdVNwUAAAAUZwUK0T/++KOysrJyjWdkZOiXX3656qYAAACA4szDpHjVqlX2n9euXSs/Pz97OysrSwkJCapWrVqhNQcAAAAUR0YhOioqSpLkcDjUu3dvl30lSpRQtWrVNH369EJrDgAAACiOjEJ0dna2JKl69erauXOnKlSocE2aAgAAAIozoxCd48iRI4XdBwAAAHDDKFCIlqSEhAQlJCToxIkT9hXqHG+99dZVNwYAAAAUVwUK0S+88IImTpyoJk2aqHLlynI4HIXdFwAAAFBsFShEL1y4UIsXL1avXr0Kux8AAACg2CvQOtGZmZlq3rx5YfcCAAAA3BAKFKKfeOIJLV26tLB7AQAAAG4IBbqd49y5c1q0aJE+//xz1a9fXyVKlHDZP2PGjEJpDgAAACiOCnQl+ttvv1XDhg3l5uampKQkff311/Zjz549hdrgL7/8okcffVTly5dXyZIlFRoaql27dtn7LctSbGysKleurJIlSyoiIkKHDx92OUZqaqp69uwpX19f+fv7q1+/fjpz5kyuc2rVqpW8vb0VHBysqVOn5uplxYoVql27try9vRUaGqpPP/20UM8VAAAAN4YCXYnesGFDYfeRp1OnTqlFixZq166dPvvsM1WsWFGHDx9W2bJl7ZqpU6dqzpw5WrJkiapXr67x48crMjJS3333nby9vSVJPXv21K+//qr4+HidP39effv21YABA+xbUpxOpzp06KCIiAgtXLhQe/fu1eOPPy5/f38NGDBAkrRt2zb16NFDU6ZMUdeuXbV06VJFRUVp9+7dqlev3nV5PQAAAFA8OCzLsoq6iUsZM2aMtm7dqi1btuS537IsBQUFacSIEXrmmWckSWlpaQoICNDixYsVHR2t/fv3KyQkRDt37lSTJk0kSXFxcercubN+/vlnBQUFacGCBXruueeUnJwsT09Pe+6VK1fqwIEDkqTu3bsrPT1dq1evtudv1qyZGjZsqIULF+brfJxOp/z8/JSWliZfX98Cvy4F1XjkO9d9TgDXR+K0x4q6BQC4KeQ3rxXodo527drpnnvuueSjsKxatUpNmjTRQw89pEqVKumuu+7Sv/71L3v/kSNHlJycrIiICHvMz89PYWFh2r59uyRp+/bt8vf3twO0JEVERMjNzU07duywa1q3bm0HaEmKjIzUwYMHderUKbvm4nlyanLmyUtGRoacTqfLAwAAADe+AoXohg0bqkGDBvYjJCREmZmZ2r17t0JDQwutuf/+979asGCBatasqbVr12rgwIF6+umntWTJEklScnKyJCkgIMDleQEBAfa+5ORkVapUyWW/h4eHypUr51KT1zEunuNSNTn78zJlyhT5+fnZj+DgYKPzBwAAQPFUoHuiZ86cmef4hAkTcn1g72pkZ2erSZMmmjx5siTprrvuUlJSkhYuXKjevXsX2jzXytixYzV8+HB72+l0EqQBAABuAgW6En0pjz76qN56661CO17lypUVEhLiMlanTh0dPXpUkhQYGChJSklJcalJSUmx9wUGBurEiRMu+y9cuKDU1FSXmryOcfEcl6rJ2Z8XLy8v+fr6ujwAAABw4yvUEL19+3Z7RYzC0KJFCx08eNBl7NChQ6pataokqXr16goMDFRCQoK93+l0aseOHQoPD5ckhYeH6/Tp00pMTLRr1q9fr+zsbIWFhdk1mzdv1vnz5+2a+Ph41apVy14JJDw83GWenJqceQAAAHDrKNDtHA888IDLtmVZ+vXXX7Vr1y6NHz++UBqTpGHDhql58+aaPHmyHn74YX311VdatGiRFi1aJElyOBwaOnSoJk2apJo1a9pL3AUFBSkqKkrSn1euO3bsqP79+2vhwoU6f/68Bg0apOjoaAUFBUmSHnnkEb3wwgvq16+fRo8eraSkJM2ePdvltpUhQ4aoTZs2mj59urp06aIPPvhAu3btsnsBAADAraNAIdrPz89l283NTbVq1dLEiRPVoUOHQmlMkpo2baqPP/5YY8eO1cSJE1W9enXNmjVLPXv2tGtGjRql9PR0DRgwQKdPn1bLli0VFxfnckX8/fff16BBg9S+fXu5ubmpW7dumjNnjsv5rFu3TjExMWrcuLEqVKig2NhYe41oSWrevLmWLl2qcePG6dlnn1XNmjW1cuVK1ogGAAC4BRXrdaJvNqwTDeBaYZ1oACgc+c1rBboSnSMxMVH79++XJNWtW1d33XXX1RwOAAAAuCEUKESfOHFC0dHR2rhxo/z9/SVJp0+fVrt27fTBBx+oYsWKhdkjAAAAUKwUaHWOwYMH6/fff9e+ffuUmpqq1NRUJSUlyel06umnny7sHgEAAIBipUBXouPi4vT555+rTp069lhISIjmz59fqB8sBAAAAIqjAl2Jzs7OVokSJXKNlyhRQtnZ2VfdFAAAAFCcFShE33PPPRoyZIiOHz9uj/3yyy8aNmyY2rdvX2jNAQAAAMVRgUL0vHnz5HQ6Va1aNd1555268847Vb16dTmdTs2dO7ewewQAAACKlQLdEx0cHKzdu3fr888/14EDByT9+c2AERERhdocAAAAUBwZXYlev369QkJC5HQ65XA4dO+992rw4MEaPHiwmjZtqrp162rLli3XqlcAAACgWDAK0bNmzVL//v3z/PYWPz8/Pfnkk5oxY0ahNQcAAAAUR0Yh+ptvvlHHjh0vub9Dhw5KTEy86qYAAACA4swoRKekpOS5tF0ODw8PnTx58qqbAgAAAIozoxB92223KSkp6ZL7v/32W1WuXPmqmwIAAACKM6MQ3blzZ40fP17nzp3Lte/s2bN6/vnn1bVr10JrDgAAACiOjJa4GzdunP7zn//ob3/7mwYNGqRatWpJkg4cOKD58+crKytLzz333DVpFAAAACgujEJ0QECAtm3bpoEDB2rs2LGyLEuS5HA4FBkZqfnz5ysgIOCaNAoAAAAUF8ZftlK1alV9+umnOnXqlL7//ntZlqWaNWuqbNmy16I/AAAAoNgp0DcWSlLZsmXVtGnTwuwFAAAAuCEYfbAQAAAAACEaAAAAMEaIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADAECEaAAAAMESIBgAAAAwRogEAAABDhGgAAADA0A0Vol9++WU5HA4NHTrUHjt37pxiYmJUvnx5+fj4qFu3bkpJSXF53tGjR9WlSxeVKlVKlSpV0siRI3XhwgWXmo0bN6pRo0by8vJSjRo1tHjx4lzzz58/X9WqVZO3t7fCwsL01VdfXYvTBAAAQDF3w4TonTt36vXXX1f9+vVdxocNG6b/+7//04oVK7Rp0yYdP35cDzzwgL0/KytLXbp0UWZmprZt26YlS5Zo8eLFio2NtWuOHDmiLl26qF27dtqzZ4+GDh2qJ554QmvXrrVrli1bpuHDh+v555/X7t271aBBA0VGRurEiRPX/uQBAABQrDgsy7KKuokrOXPmjBo1aqTXXntNkyZNUsOG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\n",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Visualize the distribution of target variable\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.countplot(data=df, x='RainTomorrow')\n",
    "plt.title('Distribution of Rain Tomorrow')\n",
    "plt.xlabel('Rain Tomorrow')\n",
    "plt.ylabel('Count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "b74b4714",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>MinTemp</th>\n",
       "      <th>MaxTemp</th>\n",
       "      <th>Rainfall</th>\n",
       "      <th>WindGustSpeed</th>\n",
       "      <th>Humidity9am</th>\n",
       "      <th>Humidity3pm</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>145460.000000</td>\n",
       "      <td>145460.000000</td>\n",
       "      <td>145460.000000</td>\n",
       "      <td>145460.000000</td>\n",
       "      <td>145460.000000</td>\n",
       "      <td>145460.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>12.194034</td>\n",
       "      <td>23.221348</td>\n",
       "      <td>2.360918</td>\n",
       "      <td>40.035230</td>\n",
       "      <td>68.880831</td>\n",
       "      <td>51.539116</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>6.365750</td>\n",
       "      <td>7.088124</td>\n",
       "      <td>8.382488</td>\n",
       "      <td>13.118253</td>\n",
       "      <td>18.854765</td>\n",
       "      <td>20.471189</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>-8.500000</td>\n",
       "      <td>-4.800000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>7.700000</td>\n",
       "      <td>18.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>57.000000</td>\n",
       "      <td>37.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>12.100000</td>\n",
       "      <td>22.700000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>39.000000</td>\n",
       "      <td>69.000000</td>\n",
       "      <td>51.539116</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>16.800000</td>\n",
       "      <td>28.200000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>46.000000</td>\n",
       "      <td>83.000000</td>\n",
       "      <td>65.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>33.900000</td>\n",
       "      <td>48.100000</td>\n",
       "      <td>371.000000</td>\n",
       "      <td>135.000000</td>\n",
       "      <td>100.000000</td>\n",
       "      <td>100.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             MinTemp        MaxTemp       Rainfall  WindGustSpeed  \\\n",
       "count  145460.000000  145460.000000  145460.000000  145460.000000   \n",
       "mean       12.194034      23.221348       2.360918      40.035230   \n",
       "std         6.365750       7.088124       8.382488      13.118253   \n",
       "min        -8.500000      -4.800000       0.000000       6.000000   \n",
       "25%         7.700000      18.000000       0.000000      31.000000   \n",
       "50%        12.100000      22.700000       0.000000      39.000000   \n",
       "75%        16.800000      28.200000       1.000000      46.000000   \n",
       "max        33.900000      48.100000     371.000000     135.000000   \n",
       "\n",
       "         Humidity9am    Humidity3pm  \n",
       "count  145460.000000  145460.000000  \n",
       "mean       68.880831      51.539116  \n",
       "std        18.854765      20.471189  \n",
       "min         0.000000       0.000000  \n",
       "25%        57.000000      37.000000  \n",
       "50%        69.000000      51.539116  \n",
       "75%        83.000000      65.000000  \n",
       "max       100.000000     100.000000  "
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Summary Statistics of Selected Numerical Features\n",
    "\n",
    "selected_numerical_features = ['MinTemp','MaxTemp','Rainfall','WindGustSpeed','Humidity9am','Humidity3pm']\n",
    "df[selected_numerical_features].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ed71b7f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAt0AAAJwCAYAAABVtd3CAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAADGhElEQVR4nOzddXiTVxsG8DuppO5u1A0rUGAFChQKHTCGDx0uwwYUBhSXjW6MMYY7ZVgZziguxUdxb3Gvu1BNvj/6EQhNgUBDhft3Xe+15eSc857z0rQnT573RCCRSCQgIiIiIiKlEZb2AIiIiIiIKjouuomIiIiIlIyLbiIiIiIiJeOim4iIiIhIybjoJiIiIiJSMi66iYiIiIiUjItuIiIiIiIl46KbiIiIiEjJuOgmIiIiIlIyLrqJ6Itjb2+P3r17K/UcjRs3RuPGjZV6jvIqPDwcAoEA4eHhJdqvQCDAtGnTSrRPIqKSwkU3EZV5ISEhEAgE0kNDQwOurq4YNmwYYmNjS3t49Bnt3buXC2siKpdUS3sAREQfasaMGXBwcEB2djZOnTqFJUuWYO/evbhx4wa0tLQ+uJ+oqCgIhcqNORw8eFCp/X+p9u7di0WLFsldeL98+RKqqvyzRkRlE387EVG50aJFC3h7ewMA+vfvD2NjY8ydOxe7du1C165dP7gfkUj03jqZmZnQ1tb+6LGqq6t/dNuyKisrS+6bm/z8fIjF4lKfs4aGRqmen4joXZheQkTlVpMmTQAADx8+BADMmTMH9erVg7GxMTQ1NVGrVi1s3bq1SLu3c7pfpa8cP34cQ4YMgZmZGWxsbHDt2jUIBALs3r1bWvfixYsQCASoWbOmTJ8tWrRA3bp1pY/l5XQvWLAAlStXhpaWFgwNDeHt7Y2NGzfK1Hn+/Dn69u0Lc3NziEQiVK5cGatXr/7ga7J+/XrUqVNHeo6GDRsWibovXrwYlStXhkgkgpWVFYYOHYqUlBSZOo0bN0aVKlVw8eJFNGzYEFpaWpgwYQIePXoEgUCAOXPmYN68eXBycoJIJMKtW7cAAJGRkejYsSOMjIygoaEBb29vmetXnJMnT6JTp06ws7ODSCSCra0tRo0ahZcvX0rr9O7dG4sWLQIAmXSjV+TldF++fBktWrSAnp4edHR00LRpU/z3338ydV79+58+fRqBgYEwNTWFtrY22rVrh/j4+PeOnYjoQzDSTUTl1v379wEAxsbGAIC//voL3377Lbp3747c3FyEhoaiU6dO2LNnD1q1avXe/oYMGQJTU1NMmTIFmZmZqFKlCgwMDHDixAl8++23AAoXh0KhEFevXkVaWhr09PQgFotx5swZDBw4sNi+V6xYgR9//BEdO3bEiBEjkJ2djWvXruHcuXPo1q0bACA2NhZfffUVBAIBhg0bBlNTU+zbtw/9+vVDWloaRo4c+c7xT58+HdOmTUO9evUwY8YMqKur49y5czh69CiaN28OAJg2bRqmT58Of39/DB48GFFRUViyZAnOnz+P06dPQ01NTdpfYmIiWrRogS5duqBHjx4wNzeXPrdmzRpkZ2dj4MCBEIlEMDIyws2bN1G/fn1YW1tj/Pjx0NbWxj///IO2bdti27ZtaNeuXbFj37JlC7KysjB48GAYGxsjIiICCxYswLNnz7BlyxYAwKBBg/DixQscOnQI69ate/c/JoCbN2/C19cXenp6GDt2LNTU1LBs2TI0btwYx48fl3mTBADDhw+HoaEhpk6dikePHmHevHkYNmwYNm/e/N5zERG9l4SIqIxbs2aNBIDk8OHDkvj4eMnTp08loaGhEmNjY4mmpqbk2bNnEolEIsnKypJpl5ubK6lSpYqkSZMmMuWVKlWS9OrVq0j/DRo0kOTn58vUbdWqlaROnTrSx+3bt5e0b99eoqKiItm3b59EIpFILl26JAEg2bVrl7Reo0aNJI0aNZI+btOmjaRy5crvnGe/fv0klpaWkoSEBJnyLl26SPT19YvM7013796VCIVCSbt27SQFBQUyz4nFYolEIpHExcVJ1NXVJc2bN5eps3DhQgkAyerVq2XGD0CydOlSmb4ePnwoASDR09OTxMXFyTzXtGlTSdWqVSXZ2dky565Xr57ExcVFWnbs2DEJAMmxY8ekZfLmFhwcLBEIBJLHjx9Ly4YOHSop7k8XAMnUqVOlj9u2bStRV1eX3L9/X1r24sULia6urqRhw4bSslf//v7+/tJrJZFIJKNGjZKoqKhIUlJS5J6PiEgRTC8honLD398fpqamsLW1RZcuXaCjo4MdO3bA2toaAKCpqSmtm5ycjNTUVPj6+uLSpUsf1P+AAQOgoqIiU/aqfWZmJgDg1KlTaNmyJby8vHDy5EkAhdFvgUCABg0aFNu3gYEBnj17hvPnz8t9XiKRYNu2bWjdujUkEgkSEhKkR0BAAFJTU985j507d0IsFmPKlClFbhJ9lYJx+PBh5ObmYuTIkTJ1BgwYAD09PYSFhcm0E4lE6NOnj9zzdejQAaamptLHSUlJOHr0KL777jukp6dLx56YmIiAgADcvXsXz58/L3b8b/7bZWZmIiEhAfXq1YNEIsHly5eLbVecgoICHDx4EG3btoWjo6O03NLSEt26dcOpU6eQlpYm02bgwIEy6Sq+vr4oKCjA48ePFT4/EdHbmF5CROXGokWL4OrqClVVVZibm8PNzU1m8bhnzx78/PPPuHLlCnJycqTlby6k3sXBwaFIma+vL/Lz83H27FnY2toiLi4Ovr6+uHnzpsyi29PTE0ZGRsX2PW7cOBw+fBh16tSBs7Mzmjdvjm7duqF+/foAgPj4eKSkpGD58uVYvny53D7i4uKK7f/+/fsQCoXw9PQsts6rxaObm5tMubq6OhwdHYssLq2trYu9OfLta3Xv3j1IJBJMnjwZkydPLnb8r94gve3JkyeYMmUKdu/ejeTkZJnnUlNTi51TceLj45GVlVVkrgDg4eEBsViMp0+fonLlytJyOzs7mXqGhoYAUGQ8REQfg4tuIio36tSpI9295G0nT57Et99+i4YNG2Lx4sWwtLSEmpoa1qxZU+RmxeK8GW19xdvbGxoaGjhx4gTs7OxgZmYGV1dX+Pr6YvHixcjJycHJkyffma8MFC70oqKisGfPHuzfvx/btm3D4sWLMWXKFEyfPh1isRgA0KNHD/Tq1UtuH9WqVfugeZQUedejuOdejX/MmDEICAiQ28bZ2VlueUFBAZo1a4akpCSMGzcO7u7u0NbWxvPnz9G7d29p38r29qccr0gkks9yfiKq2LjoJqIKYdu2bdDQ0MCBAwdktgRcs2bNJ/Wrrq6OOnXq4OTJk7Czs4Ovry+Awgh4Tk4ONmzYgNjYWDRs2PC9fWlra6Nz587o3LkzcnNz0b59e/zyyy8ICgqCqakpdHV1UVBQAH9/f4XH6eTkBLFYjFu3bsHLy0tunUqVKgEo3Kf8zZSL3NxcPHz48KPO+8qr/tTU1BTu5/r167hz5w7Wrl2Lnj17SssPHTpUpO6HfmphamoKLS0tREVFFXkuMjISQqEQtra2Co2TiOhTMKebiCoEFRUVCAQCFBQUSMsePXqEnTt3fnLfvr6+OHfuHI4dOyZddJuYmMDDwwO//fabtM67JCYmyjxWV1eHp6cnJBIJ8vLyoKKigg4dOmDbtm24ceNGkfbv27qubdu2EAqFmDFjRpHI8KtIrb+/P9TV1TF//nyZ6O2qVauQmpr6QTu8FMfMzAyNGzfGsmXLEB0drdD4X0WY3xyTRCLBX3/9VaTuq73T397iUF6fzZs3x65du/Do0SNpeWxsLDZu3IgGDRpAT0/vnX0QEZUkRrqJqEJo1aoV5s6di6+//hrdunVDXFwcFi1aBGdnZ1y7du2T+vb19cUvv/yCp0+fyiyuGzZsiGXLlsHe3h42Njbv7KN58+awsLBA/fr1YW5ujtu3b2PhwoVo1aoVdHV1AQC//vorjh07hrp162LAgAHw9PREUlISLl26hMOHDyMpKanY/p2dnTFx4kTMnDkTvr6+aN++PUQiEc6fPw8rKysEBwfD1NQUQUFBmD59Or7++mt8++23iIqKwuLFi1G7dm306NHjk67TokWL0KBBA1StWhUDBgyAo6MjYmNjcfbsWTx79gxXr16V287d3R1OTk4YM2YMnj9/Dj09PWzbtk1uLnWtWrUAAD/++CMCAgKgoqKCLl26yO33559/xqFDh9CgQQMMGTIEqqqqWLZsGXJycjB79uxPmisRkcJKa9sUIqIP9WpLt/Pnz7+z3qpVqyQuLi4SkUgkcXd3l6xZs0YyderUIlvMFbdlYHH9p6WlSVRUVCS6uroyWwquX79eAkDy/fffF2nz9paBy5YtkzRs2FBibGwsEYlEEicnJ8lPP/0kSU1NlWkXGxsrGTp0qMTW1laipqYmsbCwkDRt2lSyfPnyd879ldWrV0tq1KghEYlEEkNDQ0mjRo0khw4dkqmzcOFCibu7u0RNTU1ibm4uGTx4sCQ5ObnI+OVtcfhqy8Dff/9d7vnv378v6dmzp8TCwkKipqYmsba2lnzzzTeSrVu3SuvI2zLw1q1bEn9/f4mOjo7ExMREMmDAAMnVq1clACRr1qyR1svPz5cMHz5cYmpqKhEIBDL/tnhry0CJpHA7x4CAAImOjo5ES0tL4ufnJzlz5oxMneL+/eWNk4joYwkkEt4hQkRERESkTMzpJiIiIiJSMi66iYiIiIiUjItuIiIiIiIl46KbiIiIiMq1EydOoHXr1rCysoJAIPig7WLDw8NRs2ZNiEQiODs7IyQkRKlj5KKbiIiIiMq1zMxMVK9eHYsWLfqg+g8fPkSrVq3g5+eHK1euYOTIkejfvz8OHDigtDFy9xIiIiIiqjAEAgF27NiBtm3bFltn3LhxCAsLk/kysi5duiAlJQX79+9XyrgY6SYiIiKiMiUnJwdpaWkyR05OTon1f/bsWfj7+8uUBQQE4OzZsyV2jrfxGykJABCm5lbaQ6gwKvdyL+0hVBhagZNLewgVgnpOWmkPocK4oFK/tIdQYThqPyvtIVQIjk5OpXZuZa4dzk/siunTp8uUTZ06FdOmTSuR/mNiYmBubi5TZm5ujrS0NLx8+RKampolcp43cdFNRERERGVKUFAQAgMDZcpEIlEpjaZkcNFNRERERGWKSCRS6iLbwsICsbGxMmWxsbHQ09NTSpQb4KKbiIiIiD6CQE1Q2kP4aD4+Pti7d69M2aFDh+Dj46O0c/JGSiIiIiIq1zIyMnDlyhVcuXIFQOGWgFeuXMGTJ08AFKar9OzZU1r/hx9+wIMHDzB27FhERkZi8eLF+OeffzBq1CiljZGRbiIiIiJSmFC17ES6L1y4AD8/P+njV/ngvXr1QkhICKKjo6ULcABwcHBAWFgYRo0ahb/++gs2NjZYuXIlAgIClDZGLrqJiIiIqFxr3Lgx3vXVM/K+bbJx48a4fPmyEkcli4tuIiIiIlKYQI1Zyorg1SIiIiIiUjIuuomIiIiIlIzpJURERESksLJ0I2V5wEg3EREREZGSMdJNRERERAorz1+OUxoY6SYiIiIiUjJGuomIiIhIYczpVgwj3URERERESsZFNxERERGRknHRTURERESkZMzpJiIiIiKFcfcSxTDSTURERESkZIx0ExEREZHCuHuJYrjoJiIiIiKFCVS46FYE00uIiIiIiJSMi24iIiIiIiXjopuIiIiISMmY001EREREChMyp1shjHQTERERESkZI91EREREpDCBkJFuRTDSTURERESkZIx0ExEREZHCBCqM3SqCV4uIiIiISMm46CYiIiIiUjIuut+hcePGGDlyZGkPg4iIiKjMEaoIlHZURF9cTnfv3r2xdu1aDBo0CEuXLpV5bujQoVi8eDF69eqFkJAQbN++HWpqah/Ub0hICPr06fPOOg8fPoS9vf3HDv2LZdTAG46j+0G/ZhVoWJnhQochiN19pLSHVabo+rWEfkBbqOgbIvfpIyRuWo7ch3fl1rX46WdouFUtUp517QLi5s8EABh82wXatX2hYmQCSX4+ch/fR/KO9ch9eEep8yiLtu89iE07w5CUkgonezuM7N8Lnq5Ocus+fPIMqzZtRdT9h4iJT8Dwvj3wXesWn3nEZcOWA+HY8O8hJKamwcXOBqP7dEZlZ3u5dR88fYFlW/5F1IMniE5IwsieHdG1ZVOZOm2HTUR0QlKRth2aN8TYvl2VMYUyQyKRIGzzYpw+sg0vM9Ph6O6FLgMmwcyyUrFtDuxYiSvnjiD2+UOoqYvg6OaFtt1HwtzaQVpn47IZiLr+H1KT4iHS0IKDW3W07TEKFm/UKc/+/fdfbN22DcnJyXB0cMDgwYPh5uZWbP2TJ0/i73XrEBsbC2srK/Tp2xd1ateWPi+RSLBu/Xrs378fmZmZ8PT0xLChQ2FtbS3TT0REBDZu3IiHjx5BXV0dVatUwZQpU5Q2Tyo/vshIt62tLUJDQ/Hy5UtpWXZ2NjZu3Ag7OztpmZGREXR1dT+oz86dOyM6Olp6+Pj4YMCAATJltra2JT6XL4GKthbSrkXhxo/TS3soZZJW7QYw+q4vUv7djBczApH79CHMR06DUFdfbv24xb/iaWAv6fF8yjBICgqQdeG0tE5ezAskblyOF1N/RMxv45GfGAeLUdMg1NH7XNMqE46cOouFazagd+f2WPnHz3C2t8PoGb8iOSVVbv3snBxYmpth0PddYGRo8HkHW4YcOnMBf63bhn4dW2Ft8AQ4V7LBiOD5SEpNk1s/OzcX1mYmGNKtLYwN5P+MrZk1HnuX/io9Fkz8EQDQtG4tpc2jrDi0aw3C921El4GT8VPwBqiLNLHw5x+Ql5tTbJu7Ny+gYUAXjJm1HsMnL0dBfj4W/PwDcrKzpHXsHD3RY8gMTJ63E0MnLQEkEiycOQjigoLPMS2lOn78OJavWIHu3bphwYIFcHB0xKTJk5GSkiK3/q1bt/Drb78hoHlzLFywAD4+Ppg5cyYePXokrbNl61bs3r0bw4cNw7w//4SGhgYmTZ6M3NxcaZ1Tp07h9zlz0KxZMyxauBBz5sxB48aNlTvZUiQQCpR2VERf5KK7Zs2asLW1xfbt26Vl27dvh52dHWrUqCEtezu9xN7eHrNmzULfvn2hq6sLOzs7LF++HACgqakJCwsL6aGurg4tLS3pYw0NDQwaNAimpqbQ09NDkyZNcPXqVWnf06ZNg5eXF1avXg07Ozvo6OhgyJAhKCgowOzZs2FhYQEzMzP88ssvMnMRCARYsmQJWrRoAU1NTTg6OmLr1q1KunKlI/7ACdyZOg+xuw6X9lDKJP1mbZB+8iAyTh9BXvRTJK5fAkluDnQb+MutL87MQEFaivTQ9PSCJDcHmW8sujMjTiD79lXkJ8Qi78VTJG1eBaGWNtRt7D/TrMqGzbv3oXUzP7Rq2ggOtjYY80NfaIhECDtyXG59DxcnDO3dDf6+PlBX/eI+SJTaFHYEbZrUR+vG9eBoY4nx/btCQ10d/4aflVvf08keP/bogOb1ahd73Qz1dGFsoC89Tl26DhtzU9T0dFHmVEqdRCLBsbD1+LrDAFSv7QfrSq7oNewXpCbH4+r5o8W2GzZpKXz82sDK1hk29m74fuhMJCdE48mDW9I6DZp1hIunN4zNrGHn6InWXYcjOTEGifEvPsfUlGrHjh1o8fXXaN68OSrZ2WH4sGEQiUQ4ePCg3Pq7du2Cd61a6NixI+zs7NCzZ084OTnh33//BVD477Bz50506dIFPj4+cHBwwJjRo5GYmIgzZwt/rgsKCrB02TL079cPrVq1go2NDSrZ2aFhw4afbd5Utn2Ri24A6Nu3L9asWSN9vHr16vemhwDAH3/8AW9vb1y+fBlDhgzB4MGDERUV9d52nTp1QlxcHPbt24eLFy+iZs2aaNq0KZKSXn9cev/+fezbtw/79+/Hpk2bsGrVKrRq1QrPnj3D8ePH8dtvv2HSpEk4d+6cTN+TJ09Ghw4dcPXqVXTv3h1dunTB7du3FbgaVG6pqEK9khOyb71+AweJBNm3r0LkWPzHqG/SaeCPzIiTkBQXNVNRhW7DAIizMpD77GEJDLp8yMvLx537D1GrehVpmVAohHe1KrgZJT91h4C8/HxEPnyCOlXdpWVCoRC1q7rj+p0HJXaO/aci0LqxDwSCihkReyUx7jnSUhLgVvUraZmmti7snaviYdTVd7SU9TIrAwCgrSP/E7Cc7CycPbYTxmbWMDS2+LRBl7K8vDzcvXcPXl5e0jKhUAgvLy/cjoyU2+Z2ZCS83gi6AUCtWrWk9WNiYpCcnIwab/Spra0NNzc3RP7/7+29e/eQmJgIgUCAocOGoVv37pg8ebJMtLyiYU63Yr7YRXePHj1w6tQpPH78GI8fP8bp06fRo0eP97Zr2bIlhgwZAmdnZ4wbNw4mJiY4duzYO9ucOnUKERER2LJlC7y9veHi4oI5c+bAwMBAJiotFouxevVqeHp6onXr1vDz80NUVBTmzZsHNzc39OnTB25ubkXO16lTJ/Tv3x+urq6YOXMmvL29sWDBgmLHk5OTg7S0NJkjTyJ+79yp7FHR0YNARQUFaSky5QVpKVDRN3xve3UHF6jb2CP91KEiz2lW84bdwlBUWrIFes2+RczcqRBnpJfU0Mu81PR0FIjFMNKXXaQYGughsZj0EgJS0jL+f91k00SM9PWQlCI/vURRx89fRUbmS7Rq5FMi/ZVlaSkJAAA9A2OZcl0DY6SlJH5QH2KxGNtCZsPRrQas7GQ/GThxIBSjetRF4Pdf4dblUxg+eTlUP/BeprIqLS0NYrEYhoayvwMNDQyQnFT0vgAASE5OhqGBQdH6ycnS5wHI7/P/z0XHxAAANmzYgK5dumD6tGnQ0dHBuPHjkZ7+5fzupOJ9sYtuU1NTtGrVCiEhIVizZg1atWoFExOT97arVq2a9P8FAgEsLCwQFxf3zjZXr15FRkYGjI2NoaOjIz0ePnyI+/fvS+vZ29vL5JCbm5vD09MTQqFQpuzt8/n4+BR5/K5Id3BwMPT19WWOf8TyfxFRxabbwB+5zx7JvekyO/I6XswYiZhfx+HljUswHTS22Dxxos9p97HT8PGqDFMjg9IeSomLOBmGUT3qSo+C/PxP7nPzyl/w4uk99B31W5HnajdohaDf/8HI6athZlkJq+aOeWeuOBVPIi4MXnXu0gUNGjSAi4sLRgUGAii8SZPoy006RGGKybBhwwAAixYt+qA2b+9mIhAIIBa/O0qckZEBS0tLhIeHF3nO4I131vL6/pjzvU9QUBAC//+L4JWjRhX/ZqSKqCAjDZKCAqjoGciUq+gZoCA1+Z1tBeoiaNf2RfKujXKfl+TmID8uBvlxMch5cAfWvyyBbgN/pO7bVlLDL9P0dXWhIhQiKVU2qp2ckgZjA775KI6Bns7/r5tsVDspNQ1GxdwkqYjo+EScvx6JX0cP+uS+yqJq3o1h7/x6d6H8/MKb9NJSEqFvaCotT09JhI39+1PINq+chRuXTmDU9DVy00Y0tXWhqa0LM8tKcHCpjp/61MfViCPwbtCyBGZTOvT09CAUCqUR6FeSU1JgaGQkt42hoSGS37rJMjklRRrZfvXf5ORkGL3RR3JKCpwcHQFAWv7mhgzqamqwtLBAXHz8p02qjBJU0DQQZfliI90A8PXXXyM3Nxd5eXkICAhQ2nlq1qyJmJgYqKqqwtnZWeb4kOj6+/z3339FHnt4eBRbXyQSQU9PT+ZQE3zRPwrlV0Hhdn4aHq8/gYFAAA33ash58O57DbS960OgpobM/+TfFFiEQABBOf/YWRFqaqpwdXLAxWs3pWVisRgXr99AZbeKffPep1BTVYW7gx3O33j98ycWi3H+RhSqujp+cv97ws/CUF8X9WtUeX/lckhDUxtmlnbSw9LGCXoGJoi68fpenpdZGXh07zoc3KoX249EIsHmlbNwNeIoRkxdCRNzm/eeWwIJJJLCnOjyTE1NDS7OzrjyxmYFYrEYV65cgYe7u9w2Hu7uuHLlikzZ5cuXpfUtLCxgaGgo02dmVhaioqLg/v+/t84uLlBTU8PzZ8+kdfLz8xEbFwczM7OSmh6VY190pFtFRUWahqGioqK08/j7+8PHxwdt27bF7Nmz4erqihcvXiAsLAzt2rWDt7f3J/X/Kle8QYMG2LBhAyIiIrBq1aoSGn3pU9HWgrbz68iBloMN9Kq7IzcpFdlPo0txZGVD6qFdMO07AjmP7yH34V3o+beGQKSB9NOFu72Y9B2J/JREpGxfJ9NOp4E/si6fgzhTNtdQoC6CfqtOeHk1AvkpyVDR1YOuX0uoGhrL7HDyJej8bQvMmr8M7k4O8HBxwpY9+/EyOwctmzYCAPz81xKYGBnih++7ACi8+fLR///g5uXnIz4xGXcfPoKmhgZsLMv3zWmK6NqqKWYsWQsPRzt4OtsjdO9RZOfk4Jv/52BPWxQCUyMDDO3aFkDhtXr4rPC1nFdQgPikFNx59BSaGiLYWrxerIjFYuw5fhatGn4FVSX+zi5LBAIB/Fr1wP5ty2FmYQdjM2vs2bwI+oamqF67ibTeX9P7o3qdpmjconDP8s0rf8GFU/swaOxfEGloIzW5MDdcU0sH6iINJMQ+w8Uz++FRrR509AyRkhSLgztWQV1dhCo1G5TKXEtSu3bt8MfcuXBxcYGbqyt27tqFnJwcNGvWDAAwZ84cGBsbSzdQaNOmDcaOG4dt27ejTu3aOH78OO7evYsfhw8HUPjv0LZtW4SGhsLaygrm5uZYt24djI2NUe//KZ7aWlpo2bIl1q1fDxNTU5ibmUnv2/JtUP6vqTwCIQN2iviiF91A4cdQyiYQCLB3715MnDgRffr0QXx8PCwsLNCwYUOYm5t/cv/Tp09HaGgohgwZAktLS2zatAmenp4lMPKyQb9WFfgceb1g9JwzAQDw9O/tuNYvqLSGVWZknT+FJB09GLbpBhU9Q+Q+fYjYedMhTitMi1A1NgHeulFW1dwaGq6VETNXzhc2iMVQs7SBTr0mUNHRQ0FmOnIf3kX0b0HIe/H0c0ypzGjawAcpaelYFboVScmpcHaohDlTxsHo/+klsfGJMrtnJCQno2/gROnj0F1hCN0VBq/KHljw86TPPv7S0qyeN1LSMrB8yx4kpqTBtZIN5o0fLt2DOzYhCcI3rlt8Uiq+Hz9L+njDnsPYsOcwanq4YMnU16lwEdcjEZOQhNaN632+yZQBzdr0QW72S2xcNgMvs9Lh5F4DQycugZq6SFonIfYZMtNfp1OcPPgPAGDetL4yffUYMhM+fm2gqqaOe7cv4VjYemRlpEHXwBjOHrUw+ue/oasve9NmedSoUSOkpqVh/bp1SEpOhpOjI2bOmCFNE4mLj5dZMHp6emLc2LFY+/ffCAkJgbW1NSZPnizzhXadOnZEdnY25i9YgIyMDFSuXBkzZ8yAurq6tE7/fv2goqKCOXPmICcnB+5ubvg1OPiDv/ODKjaBRCKRlPYg6OMJBALs2LEDbdu2/aR+wtQ+bHs5er/KveR/fEmK0wqcXNpDqBDUc0pm1xACLqjUL+0hVBiO2s/eX4ney9FJ/jfkfg6Xmiovgl/zyCml9V1a+LkAEREREZGScdFNRERERKRkX3xOd3nH7CAiIiKiso+LbiIiIiJSWEX9unZl4aKbiIiIiBQmEHLRrQjmdBMRERERKRkj3URERESkMH45jmJ4tYiIiIiIlIyLbiIiIiIiJeOim4iIiIhIyZjTTUREREQK4+4limGkm4iIiIhIyRjpJiIiIiKF8ctxFMNINxEREREpTCAUKO34GIsWLYK9vT00NDRQt25dREREvLP+vHnz4ObmBk1NTdja2mLUqFHIzs7+qHN/CC66iYiIiKhc27x5MwIDAzF16lRcunQJ1atXR0BAAOLi4uTW37hxI8aPH4+pU6fi9u3bWLVqFTZv3owJEyYobYxcdBMRERFRuTZ37lwMGDAAffr0gaenJ5YuXQotLS2sXr1abv0zZ86gfv366NatG+zt7dG8eXN07dr1vdHxT8FFNxERERGVKTk5OUhLS5M5cnJy5NbNzc3FxYsX4e/vLy0TCoXw9/fH2bNn5bapV68eLl68KF1kP3jwAHv37kXLli1LfjKvxqS0nomIiIiowhIIhUo7goODoa+vL3MEBwfLHUdCQgIKCgpgbm4uU25ubo6YmBi5bbp164YZM2agQYMGUFNTg5OTExo3bsz0EiIiIiL6cgQFBSE1NVXmCAoKKrH+w8PDMWvWLCxevBiXLl3C9u3bERYWhpkzZ5bYOd7GLQOJiIiISGHK/HIckUgEkUj0QXVNTEygoqKC2NhYmfLY2FhYWFjIbTN58mR8//336N+/PwCgatWqyMzMxMCBAzFx4kQIhSUfl2akm4iIiIjKLXV1ddSqVQtHjhyRlonFYhw5cgQ+Pj5y22RlZRVZWKuoqAAAJBKJUsbJSDcRERERKawsfQ18YGAgevXqBW9vb9SpUwfz5s1DZmYm+vTpAwDo2bMnrK2tpXnhrVu3xty5c1GjRg3UrVsX9+7dw+TJk9G6dWvp4rukcdFNREREROVa586dER8fjylTpiAmJgZeXl7Yv3+/9ObKJ0+eyES2J02aBIFAgEmTJuH58+cwNTVF69at8csvvyhtjAKJsmLoVK6EqbmV9hAqjMq93Et7CBWGVuDk0h5ChaCek1baQ6gwLqjUL+0hVBiO2s9KewgVgqOTU6mdO6pzgNL6dtt8QGl9lxZGuomIiIhIYWUpvaQ84I2URERERERKxkg3ERERESlMoIRt9SoyXi0iIiIiIiVjpJuIiIiIFCZUYU63IhjpJiIiIiJSMi66iYiIiIiUjOklRERERKQwbhmoGEa6iYiIiIiUjJFuAsBvUSxJN9dGlvYQKoxK421KewgVgpFKXGkPocK4Fqle2kOoMK7BsbSHUCEElt4XUnLLQAXxahERERERKRkj3URERESkMOZ0K4aRbiIiIiIiJeOim4iIiIhIybjoJiIiIiJSMuZ0ExEREZHCmNOtGC66iYiIiEhh3DJQMbxaRERERERKxkg3ERERESmM6SWKYaSbiIiIiEjJuOgmIiIiIlIyLrqJiIiIiJSMOd1EREREpDDuXqIYXi0iIiIiIiVjpJuIiIiIFCfg7iWK4KKbiIiIiBTGLQMVw/QSIiIiIiIl46KbiIiIiEjJuOgmIiIiIlIy5nQTERERkcK4ZaBieLWIiIiIiJSMkW4iIiIiUhh3L1EMI91ERERERErGSDcRERERKYw53Yrh1SIiIiIiUjIuuomIiIiIlIzpJURERESkMN5IqRhGuomIiIiIlIyRbiIiIiJSGCPdimGkm4iIiIhIyRjpJiIiIiLFcctAhfBqEREREREpWYWPdPfu3Rtr167FoEGDsHTpUpnnhg4disWLF6NXr14ICQn5pPM8evQIDg4O76yzZs0a9O7d+5POU5Ho+rWEfkBbqOgbIvfpIyRuWo7ch3fl1rX46WdouFUtUp517QLi5s8EABh82wXatX2hYmQCSX4+ch/fR/KO9ch9eEep8ygvjBp4w3F0P+jXrAINKzNc6DAEsbuPlPawSp1EIkHo+tU4fGAPsjIz4OZRFQOHBsLK2uad7fbt2YFd20KRkpwEewcn9PthBFzcPAAAcbHRGNy3i9x2o8dPQz1fP5my9LRUBA7rh6TEePy9eQ+0dXRLZnKf0Y6w/di8YzeSklPg5FAJPw7sCw9Xl2Lrh586i9UbQhETFw8bKwsM7NUDX3nXlKnz+OkzLF+7Hldv3EJBgRiVbG0wPWg0zE1NpXVuRkZh1bpNuH3nHoRCIZwd7DF7+kSIRCKlzbUskEgkuHBwASIjtiDnZRos7GvCt91U6JvaF9vmwsEFuHh4kUyZgakDOv+0T8mjLbt4HelzqvCLbgCwtbVFaGgo/vzzT2hqagIAsrOzsXHjRtjZ2ZXYOaKjo6WP58yZg/379+Pw4cPSMn19/RI5V0WgVbsBjL7ri8T1S5Dz4A70/FvDfOQ0PJ80BOL01CL14xb/CoHK6x9XoY4urKb+hawLp6VleTEvkLhxOfLjYyBQV4deszawGDUNzyb8AHFG2meZV1mmoq2FtGtReBqyDd5bF72/wRdi59ZN2PvvdgwfFQQzC0uErluFmZPH4K+la6GuLn/hdvrEUYSsWIRBwwLh4uaJPTu3YObkMViwfD30DQxhbGKGleu2y7Q5tP9f7NoeihredYv0t+iv2ajk4IikxHilzFHZjp48jSWr1mLUkIHwcHXG1t1hGDv1F/y95C8YGhT9vXfjdhRmzpmHAT27wad2LRw5fgqTZ83G8j9nw6FS4e/k59Ex+HH8ZLTwb4LeXTtDS0sTj548hbqaurSfm5FRGDftF3Tr2A7DB/WDilCI+48efxHfknc1fCVunF4Hv86/QtfIBucP/IWwVf3x3egwqKoV/4bD0NwF3wxcLX0sEH4Ry4Bi8TrS51TxfzMBqFmzJmxtbbF9++s/gtu3b4ednR1q1KghLdu/fz8aNGgAAwMDGBsb45tvvsH9+/elz//999/Q0dHB3buvo7FDhgyBu7s7cnJyYGFhIT10dHSgqqoqfWxmZoZ58+bBwcEBmpqaqF69OrZu3SrtJzw8HAKBAAcOHECNGjWgqamJJk2aIC4uDvv27YOHhwf09PTQrVs3ZGVlSds1btwYw4YNw7Bhw6Cvrw8TExNMnjwZEolEWZezROg3a4P0kweRcfoI8qKfInH9Ekhyc6DbwF9ufXFmBgrSUqSHpqcXJLk5yHxj0Z0ZcQLZt68iPyEWeS+eImnzKgi1tKFuY/+ZZlW2xR84gTtT5yF21+H3V/5CSCQS7Nm1BR07f486Pg1g7+CE4aMnIDkpERFnTxXb7t8d/8D/62/QpFlL2NrZY9Cw0RBpaODIwb0AABUVFRgaGcscEWdPol4DP2hqasn0tT9sJ7IyM9CmvfzIeHmwZdcetGreFC38/WBvZ4vAIQOhIVLHvsNH5dbf9m8Y6tT0Qpf2bVDJ1gZ9e3SBi6MjdoTtl9ZZtX4T6taqgR/6fA8XJwdYW1qgft3aMov4RSvXov03LdGtYzs42NnCzsYafg3qQV1NTelzLk0SiQTXT/2Nmk1/gH3lpjC2dINf59+QlRaHRzff/foWClWgpWsqPTS1DT/TqMseXsdPJxAIlHZURF/EohsA+vbtizVr1kgfr169Gn369JGpk5mZicDAQFy4cAFHjhyBUChEu3btIBaLAQA9e/ZEy5Yt0b17d+Tn5yMsLAwrV67Ehg0boKUl+4f0bcHBwfj777+xdOlS3Lx5E6NGjUKPHj1w/PhxmXrTpk3DwoULcebMGTx9+hTfffcd5s2bh40bNyIsLAwHDx7EggULZNqsXbsWqqqqiIiIwF9//YW5c+di5cqVn3K5lEtFFeqVnJB96+rrMokE2bevQuTo9kFd6DTwR2bESUhyc4o9h27DAIizMpD77GEJDJoqotiYaKQkJ6GaVy1pmba2DlzcPBAVeVNum7y8PNy/d0emjVAoRDWvWrhTTJv7d6Pw8ME9NG3eSqb86ZNH2LJpLYYHTii3f2Ty8vJw594D1PKqJi0TCoWoWb0abkbKT+26FXkHtapXkymrXbO6tL5YLMZ/Fy7BxsoKP039Ge2+74fBY4Jw6r8Iaf3klFTcvnMXBgb6GDZ2Itp/3x8jgqbg+q3bSphl2ZKe9AxZ6fGwdqknLRNp6sLMthpiH195Z9vUhMdYN9MXG3/1x5GNY5Ce/ELJoy27eB0/nUAoVNpREX0xn4f06NEDQUFBePz4MQDg9OnTCA0NRXh4uLROhw4dZNqsXr0apqamuHXrFqpUqQIAWLZsGapVq4Yff/wR27dvx7Rp01CrVi28S05ODmbNmoXDhw/Dx8cHAODo6IhTp05h2bJlaNSokbTuzz//jPr16wMA+vXrh6CgINy/fx+Ojo4AgI4dO+LYsWMYN26ctI2trS3+/PNPCAQCuLm54fr16/jzzz8xYMCAYseTkyO7WM0pKIBIReWd8ygpKjp6EKiooCAtRaa8IC0FahbvzqMFAHUHF6jb2CNh7cIiz2lW84bpwDEQqItQkJqMmLlTIc5IL6mhUwWTkpwEADAwNJIp1zcwlD73tvS0VIjFBTAwMCzS5vnTJ3LbHDkYBhvbSnD3rCIty8vLxZ+zZ6Bn38EwNTNHbEz5/KOdmpYOsVhcJI3E0EAfT54/l9smKSVFTn0DJCenAABSUlPx8mU2Nm3bib49umBQr+6IuHQFU4LnYO4vU+FVpTKiY2IBAGs3/YMf+vSEs4M9Dh47jtGTZmD1wrmwsbIs+cmWEVnphWlImjrGMuWauibISk8otp2ZXXU07hwMA1MHZKXF4eLhRdi9pAc6Be6GuoaOUsdcFvE60uf2xSy6TU1N0apVK4SEhEAikaBVq1YwMTGRqXP37l1MmTIF586dQ0JCgjTC/eTJE+mi29DQEKtWrUJAQADq1auH8ePHv/fc9+7dQ1ZWFpo1ayZTnpubK5PeAgDVqr2O/pibm0NLS0u64H5VFhERIdPmq6++komS+fj44I8//kBBQQFU5Cykg4ODMX36dJmyETVcMbKm+3vnUhboNvBH7rNHcm+6zI68jhczRkJFRw86vs1hOmgsomf9JDdPnL48J44dwrKFf0gfT5j2q9LPmZOTg5PHj6BTl54y5etDlsPGthIaNWmu9DGUN2JxYXpcvbre6NTmGwCAs6MDbkZG4d99h+BVpTLE/0+h+yagGVr4F96Y6uLkgEtXr2PfoaMY0Kt76QxeCe5e+hcntk+VPm7RZ+k7ahfPzr2h9P+NLd1gZlcdG4Ob4MG1/XCv0/GTx1nW8TqWPH45jmK+mEU3UJhiMmzYMADAokVFbyRr3bo1KlWqhBUrVsDKygpisRhVqlRBbm6uTL0TJ05ARUUF0dHRyMzMhK7uu3cayMjIAACEhYXB2tpa5rm377BXeyMXUSAQyDx+VfbqzcDHCgoKQmBgoExZ9Ihun9SnIgoy0iApKICKnoFMuYqeAQpSk9/ZVqAugnZtXyTv2ij3eUluDvLjYpAfF4OcB3dg/csS6DbwR+q+bSU1fCrHatetL91hBChMjQAKI96GRq+jXakpybB3dJbbh66ePoRCFaSkyP6spqYkF4mYA8DZ0+HIzclGo6YBMuU3rl7Gk8cP0OnUqxSzwkVk765t0KFzD3Tp0Vfh+ZUGfT1dCIVCJKfIvrFNTkmFkYGB3DZGBgZy6qfA0NBA2qeKigrsbW1l6tjZ2OD6rUgAgPH/69rbyn46ZmdrjdiE4qOU5VElTz90tHsdkCnIL/yb9DIjEdp6ZtLyl+kJMLbyKNK+OCJNPeib2CM18XHJDbYM43Wk0lYxk2aK8fXXXyM3Nxd5eXkICJD9A5iYmIioqChMmjQJTZs2hYeHB5KTiy4Az5w5g99++w3//vsvdHR0pIv4d/H09IRIJMKTJ0/g7Owsc9i+9UflY5w7d07m8X///QcXFxe5UW6gcKGvp6cnc3yu1BIAQEHhdn4aHm/kdAoE0HCvhpwHUe9squ1dHwI1NWT+d/yd9d7sV1DBb6qiD6eppQVLKxvpYWtnDwNDI1y/eklaJysrE3ejbsPNvbLcPtTU1ODk7IrrVy5Ky8RiMa5duQRXOW2OHtwL77r1oa9vIFP+08QZ+GPBKvyxYCX+WLASg3/8CQDw8+z5aPFNuxKY7eehpqYGV2dHXLp6XVomFotx6dp1VHZ3ldvG090Vl65dlym7eOWatL6amhrcXZzw9K30lGcvXsDcrPATSgtzM5gYGeLpc9m0nGfPo2W2FKwI1DV0oG9SSXoYmjtDS9cUz++eldbJzc5A3NNrMK/k9cH95uVkIi3xKbR0K9b1Kg6vI5W2LyrSraKigtu3b0v//02GhoYwNjbG8uXLYWlpiSdPnhRJHUlPT8f333+PH3/8ES1atICNjQ1q166N1q1bo2PH4j9S0tXVxZgxYzBq1CiIxWI0aNAAqampOH36NPT09NCrV69PmteTJ08QGBiIQYMG4dKlS1iwYAH++OOP9zcsRamHdsG07wjkPL6H3Id3oeffGgKRBtJPF94xbtJ3JPJTEpGyfZ1MO50G/si6fA7iTNk8bYG6CPqtOuHl1QjkpyRDRVcPun4toWpoLLPDyZdMRVsL2s6vt8jUcrCBXnV35CalIvtp9DtaVlwCgQDftOmEraF/w9LKBmYWFti0bjUMjYxRx6eBtN60CaNQx8cXLVu3BwC0bvcdFswNhpOLO1xc3bFn11bkZL9Ek2YtZPqPfvEMt25cxcRpvxU5t4Wl7KdeaWmFkV8b20rlbp/uTm2+wa/zFsHV2Um6ZWB2dg6+blqY9jHrzwUwNTKSpnx0aN0KIydMxT87/sVXtWvi6InTiLp3H6OHDpL22bndt5jx+5+oVtkTNapWRsSlKzgTcRHzZk0DUPhv17ldG4Rs2gwnh0pwdrDHgaPH8eT5c0wbP/qzX4PPSSAQoGqDnrh0dCn0Teyha2SNCwfnQ0vPDPaVX+8A9e/y3nCo7I8q9XsAAM7u+Q2VPPyga2iFzLQ4XDi0EAKhEM5e35TWVEoVryN9bl/UohsA9PT05JYLhUKEhobixx9/RJUqVeDm5ob58+ejcePG0jojRoyAtrY2Zs2aBQCoWrUqZs2ahUGDBsHHx6dI6sibZs6cCVNTUwQHB+PBgwcwMDBAzZo1MWHChE+eU8+ePfHy5UvUqVMHKioqGDFiBAYOHPjJ/SpT1vlTSNLRg2GbblDRM0Tu04eInTcd4v8vPFSNTQCJbBqNqrk1NFwrI2bulKIdisVQs7SBTr0mUNHRQ0FmOnIf3kX0b0HIe/H0c0ypzNOvVQU+R16/ifGcU/iz9/Tv7bjWL6i0hlXq2nbsiuzsl1i6YA4yMzPg7lkVk2f+LrNHd0z0C6SnvU6HqN+wCVJTUxC6fjVSkpPg4OiMSTN+L5JecvTQXhibmKJ6zdqfbT6loYlvfaSmpiFk4+bCL8dxtMdv0ybC6P8pIHHxCRC+cd9JFQ83TBo9Aqs3bMLKdRthbWWJmRPGSvfoBgBfn7oYNXggNm7dgQUrVsPW2grTx49BVc/XH/t3bNMKuXm5WLRqLdLTM+DkUAlzZkyGtaXFZ5t7aaneuD/ycl/ixLYpyM1Og4V9LbTst0Jmb+m0xCfIznz9iW1maiyObByN7KwUaOoYwcK+FtoO2wxNnaJpUV8KXsdPVMZ2GVm0aBF+//13xMTEoHr16liwYAHq1KlTbP2UlBRMnDgR27dvR1JSEipVqoR58+ahZcuWShmfQFLWN3Smd2rcuDG8vLwwb968T+rnUf82JTMgws21kaU9hAqj0u0PTCOidzIqiCvtIVQYoZFFvxmXqDQFtim9mxkTZygvwGc8ZblC9Tdv3oyePXti6dKlqFu3LubNm4ctW7YgKioKZmZmRern5uaifv36MDMzw4QJE2BtbY3Hjx/DwMAA1atXL6lpyPjiIt1ERERE9OnK0u4lc+fOxYABA6TfwbJ06VKEhYVh9erVcneaW716NZKSknDmzBnpphX29vZKHWPZ+lyAiIiIiMoFgUCotCMnJwdpaWkyx9vfMfJKbm4uLl68CH//17n4QqEQ/v7+OHv2rNw2u3fvho+PD4YOHQpzc3NUqVIFs2bNQkFBgVKuFcBFd7kXHh7+yaklRERERGVJcHAw9PX1ZY7g4GC5dRMSElBQUABzc3OZcnNzc8TExMht8+DBA2zduhUFBQXYu3cvJk+ejD/++AM///xzic/lFaaXEBEREVGZIu87Rd7+bpNPIRaLYWZmhuXLl0NFRQW1atXC8+fP8fvvv2Pq1Knv7+AjcNFNRERERGWKSCT64EW2iYkJVFRUEBsbK1MeGxsLCwv5uxlZWlpCTU1NZgtpDw8PxMTEIDc3F+rq6h8/+GIwvYSIiIiIFCcUKO9QgLq6OmrVqoUjR45Iy8RiMY4cOQIfHx+5berXr4979+7JfMv3nTt3YGlpqZQFN8BFNxERERGVc4GBgVixYgXWrl2L27dvY/DgwcjMzJTuZtKzZ08EBb3+TorBgwcjKSkJI0aMwJ07dxAWFoZZs2Zh6NChShsj00uIiIiISGGCMvTlOJ07d0Z8fDymTJmCmJgYeHl5Yf/+/dKbK588eQLhG+O1tbXFgQMHMGrUKFSrVg3W1tYYMWIExo0bp7QxctFNREREROXesGHDMGzYMLnPhYeHFynz8fHBf//9p+RRvcZFNxEREREprCx9OU55UHY+FyAiIiIiqqC46CYiIiIiUjKmlxARERGR4gSM3SqCV4uIiIiISMkY6SYiIiIihfFGSsUw0k1EREREpGSMdBMRERGR4srQl+OUB7xaRERERERKxkU3EREREZGSMb2EiIiIiBQmEPBGSkUw0k1EREREpGSMdBMRERGR4ngjpUJ4tYiIiIiIlIyRbiIiIiJSGL8cRzGMdBMRERERKRkX3URERERESsZFNxERERGRkjGnm4iIiIgUJ2DsVhFcdBMRERGR4ngjpUL4FoWIiIiISMkY6SYiIiIihQmYXqIQXi0iIiIiIiXjopuIiIiISMmYXkIAAK3AyaU9hAqj0nib0h5ChfHYo1FpD6FCsLgcUtpDqDAGWuwp7SFUGIKCvNIeQgXRvrQHQB+Ii24iIiIiUhx3L1EI00uIiIiIiJSMkW4iIiIiUphAyNitIrjoJiIiIiLFCZheogi+RSEiIiIiUjIuuomIiIiIlIyLbiIiIiIiJWNONxEREREpjjdSKoRXi4iIiIhIyRjpJiIiIiLFcfcShTDSTURERESkZIx0ExEREZHC+OU4iuHVIiIiIiJSMi66iYiIiIiUjOklRERERKQ4AWO3iuDVIiIiIiJSMka6iYiIiEhxQm4ZqAhGuomIiIiIlIyRbiIiIiJSmIA53Qrh1SIiIiIiUjIuuomIiIiIlIyLbiIiIiIiJWNONxEREREpjruXKISLbiIiIiJSHG+kVAivFhERERGRkjHSTURERESKEzC9RBGMdBMRERFRubdo0SLY29tDQ0MDdevWRURExAe1Cw0NhUAgQNu2bZU6Pi66iYiIiKhc27x5MwIDAzF16lRcunQJ1atXR0BAAOLi4t7Z7tGjRxgzZgx8fX2VPkYuuomIiIioXJs7dy4GDBiAPn36wNPTE0uXLoWWlhZWr15dbJuCggJ0794d06dPh6Ojo9LHyEU3ERERESlOKFTakZOTg7S0NJkjJydH7jByc3Nx8eJF+Pv7vzE0Ifz9/XH27Nlihz9jxgyYmZmhX79+JX5p5OGim4iIiIjKlODgYOjr68scwcHBcusmJCSgoKAA5ubmMuXm5uaIiYmR2+bUqVNYtWoVVqxYUeJjLw4X3UrUuHFjjBw5UqE2kZGR+Oqrr6ChoQEvL68PajNt2jSZur1791b6zQBERET0hRMIlXYEBQUhNTVV5ggKCiqRYaenp+P777/HihUrYGJiUiJ9fghuGViM3r17Y+3atQAAVVVV2NjYoFOnTpgxYwY0NDQ+qI/t27dDTU1NofNOnToV2traiIqKgo6OjsLjLu+27z2ITTvDkJSSCid7O4zs3wuerk5y6z588gyrNm1F1P2HiIlPwPC+PfBd6xafecSlQyKRIHT9ahw+sAdZmRlw86iKgUMDYWVt8852+/bswK5toUhJToK9gxP6/TACLm4eAIC42GgM7ttFbrvR46ehnq+fTFl6WioCh/VDUmI8/t68B9o6uiUzuXLAqIE3HEf3g37NKtCwMsOFDkMQu/tIaQ+rTNm27zA27tqHpJRUONvbYVS/HvB0kZ8z+eDJc6wM3Y6oB48QE5+IH/t0RedvAmTq7Nh/FDsOHEV0fAIAwMHWGn06tYFPzWpKn0tp++fwafy9NxyJqelwsbXE2O/boYqTndy624/9h7DTF3H/WWF0z8PeBkM7tZDWz8svwJJt+3DqaiSexyVCR0sTdSu7YPh3LWFqqP/Z5lRaNh85i7/3nUBiagZc7Swwtvu3qOJoK7fu9uMR2HP6Mu4/f3UtrTGsQ4BM/akrt+Df05dk2vlUccGi0X2VN4myRInfSCkSiSASiT6oromJCVRUVBAbGytTHhsbCwsLiyL179+/j0ePHqF169bSMrFYDKBwzRcVFQUnJ/lrj0/BSPc7fP3114iOjsaDBw/w559/YtmyZZg6deoHtzcyMoKurmILkfv376NBgwaoVKkSjI2NFR1yuXbk1FksXLMBvTu3x8o/foazvR1Gz/gVySmpcutn5+TA0twMg77vAiNDg8872FK2c+sm7P13OwYNHY3guUuhoaGBmZPHIDdXfr4bAJw+cRQhKxbhu2698Pv8Fajk4ISZk8cgNSUZAGBsYoaV67bLHJ2794GGpiZqeNct0t+iv2ajkoPybzwpi1S0tZB2LQo3fpxe2kMpkw6fPocFIaHo+11brP59Opwr2SJw5hwkp6bJrZ+TmwMrc1MM7tEJxgbyF36mxob4oUcnrJ49DatmT0OtKh4Y/9tfePDkuTKnUuoO/ncFczfuxsC2zbBhxki42llh2O8rkJSWLrf+xcj7CPjKC8uCfsCaKcNhbqyPob8vR1xS4e/R7NxcRD56jv5t/LFh5ijM+bEXHkXHYdSfaz7ntErFgXPXMDc0DAPbNMXGacPgYmuJoX+sRlJahtz6FyMf4OuvqmH5uAEImTQY5kYGGDJnNeKSZf8m1avqioPzJkiP4B+6fo7p0BvU1dVRq1YtHDnyOvghFotx5MgR+Pj4FKnv7u6O69ev48qVK9Lj22+/hZ+fH65cuQJbW/lvxD4VF93vIBKJYGFhAVtbW7Rt2xb+/v44dOgQACAxMRFdu3aFtbU1tLS0ULVqVWzatEmm/dvpJfb29pg1axb69u0LXV1d2NnZYfny5dLnBQIBLl68iBkzZkAgEGDatGkAgHHjxsHV1RVaWlpwdHTE5MmTkZeXp/T5f26bd+9D62Z+aNW0ERxsbTDmh77QEIkQduS43PoeLk4Y2rsb/H19oK765XxoI5FIsGfXFnTs/D3q+DSAvYMTho+egOSkREScPVVsu393/AP/r79Bk2YtYWtnj0HDRkOkoYEjB/cCAFRUVGBoZCxzRJw9iXoN/KCpqSXT1/6wncjKzECb9vIj4xVd/IETuDN1HmJ3HS7toZRJm/89gNb+jdCqiS8cbK3x06BeEInUsefICbn1PZwdMaxXF/g3+ApqavJfyw1q10C9WtVha2UBOysLDOreEZoaGrh5554yp1Lq1u8/jnaN6+LbhnXgaG2BCb07QEOkhl3Hz8ut/8vg7vjOvz7cKlnDwcoMk/t9B4lYgohbdwEAulqaWDxuEJrX9YK9pRmqOlfCuJ7tcPvRM0QnJH/OqX12Gw6eRLuGtdHG1xuO1uaY2LMtNNTVsevkBbn1fxnUBd818YGbnRUcLM0wpU97SCQSRNy6L1NPXVUVJvq60kNPW/NzTIfeEhgYiBUrVmDt2rW4ffs2Bg8ejMzMTPTp0wcA0LNnT2l6ioaGBqpUqSJzGBgYQFdXF1WqVIG6urpSxshF9we6ceMGzpw5I/2HyM7ORq1atRAWFoYbN25g4MCB+P7779+7Efsff/wBb29vXL58GUOGDMHgwYMRFRUFAIiOjkblypUxevRoREdHY8yYMQAAXV1dhISE4NatW/jrr7+wYsUK/Pnnn8qd8GeWl5ePO/cfolb1KtIyoVAI72pVcDPqbimOrOyJjYlGSnISqnnVkpZpa+vAxc0DUZE35bbJy8vD/Xt3ZNoIhUJU86qFO8W0uX83Cg8f3EPT5q1kyp8+eYQtm9ZieOAECPhtZPSWvLx8RN1/hNrVPKVlha/lyrhx5/47Wn64ggIxDp/6D9nZOaji5lwifZZFefn5iHz0HHUqu0rLhEIh6ni64Pq9xx/UR3ZOLvILCqCnrVVsnYysbAgEAuhW4MViXn4+bj96gbqVX/+8CIVC1PV0wrV7Tz6oj+ycvP9fS9nrdCHyAZr++DPaBf2BWX/vREpGZomOnT5M586dMWfOHEyZMgVeXl64cuUK9u/fL7258smTJ4iOji7VMX454cGPsGfPHujo6CA/Px85OTkQCoVYuHAhAMDa2lq6KAaA4cOH48CBA/jnn39Qp06dYvts2bIlhgwZAqAwgv3nn3/i2LFjcHNzg4WFBVRVVaGjoyOTgzRp0iTp/9vb22PMmDEIDQ3F2LFjP2peOTk5RbbdycnNhUhJ7+w+RGp6OgrEYhjpy360bGigh8fPX5TSqMqmlOQkAICBoZFMub6BofS5t6WnpUIsLoCBgWGRNs+fyv+Dc+RgGGxsK8Hd8/Uboby8XPw5ewZ69h0MUzNzxMbw34Zkpbx6Lb+VJmKkr4cnzz/tD979x08xaMLPyM3Ng6aGCLPGDoeDrfUn9VmWpaRnokAshrGe7P09xvq6eBT97i/8eGX+5jCYGOqjbmUXuc/n5OZh/j9hCPjKCzqaH3a/UnmUkp5V+HP51rU00tfFo5j4D+pj/pZ9MDXQk1m416vqiia1KsPKxAjP4hOxcNtBDJ8bgpBJg6Ei/ALimoKyNcdhw4Zh2LBhcp8LDw9/Z9uQkJCSH9BbuOh+Bz8/PyxZsgSZmZn4888/oaqqig4dOgAo3FB91qxZ+Oeff/D8+XPk5uYiJycHWlrFRxMAoFq11zf9CAQCWFhYvPfbkjZv3oz58+fj/v37yMjIQH5+PvT09D56XsHBwZg+XTYXdcyQAfhp6MCP7pOU58SxQ1i28A/p4wnTflX6OXNycnDy+BF06tJTpnx9yHLY2FZCoybNlT4GorfZWVkiZM4MZGS9xLGz5/HLwpVYOGN8hV54f4o1/x7FwXNXsDxoMETqRW/qz8svwPhF6yCRAEG9O5TCCMuPNWHhOBBxDcvHDYDojQ0SAupWl/6/i60FXGws8e2433Eh8gHqelbcT2Ho43DR/Q7a2tpwdi580axevRrVq1fHqlWr0K9fP/z+++/466+/MG/ePFStWhXa2toYOXIkcnNz39nn27uZCAQC6R2z8pw9e1b6bUkBAQHQ19dHaGgo/vjjj2LbvE9QUBACAwNlylIf3Pjo/kqCvq4uVIRCJKXK3qCSnJJW7I1VX4radetLdxgBIM3nT0lOgqHR65ttU1OSYe8o/5e8rp4+hEIVpKTI5mympiQXiZgDwNnT4cjNyUajprI7SNy4ehlPHj9Ap1Ov8uwlAIDeXdugQ+ce6NLjC7ljn4pl8Oq1/NYN0EmpaUWi34pSU1OFjWXhR8XuTvaIvPcQW8IOYewPvT+p37LKQFcbKkIhEt+60S8xNR0m+u8OvPy9NxwhYUexZOwguNhZFXn+1YI7OiEZS8f/UKGj3ABgoKtV+HP51rVMSk2Hsd67Nzz4e98JrAk7jqU/9YOrreU769qYGcFARxtPYxO/jEU3UwwVwkX3BxIKhZgwYQICAwPRrVs3nD59Gm3atEGPHj0AFN4le+fOHXh6er6nJ8WcOXMGlSpVwsSJE6Vljx9/WC5fceRtw5NdiqklQOEfU1cnB1y8dhMN63oDKLymF6/fQPsWX3ZUVVNLC5pvfIIikUhgYGiE61cvwcGp8CPjrKxM3I26jYCWbeT2oaamBidnV1y/chF1fXwBFF7fa1cuocU37YrUP3pwL7zr1oe+voFM+U8TZyD3jdSke3cjsWjeb/h59nxYWDLaSIWvZTcne1y4fgsN6xbeQyAWi3Hx2i10aNG0RM8llkiQWwFvKn9FTVUV7vbWOH/zLvxqFaZ5icVinL91D9/51y+23dqwY1i1+wgW/TQAnnK2w3u14H4aE49lQYNhoKuttDmUFWqqqvCwt0LErfvwq1kZQOG1jLh9H52bFt3d4pWQvcexes8xLBzdF54O796SFQBik1KRmpkFU4MvZwtV+nBcdCugU6dO+Omnn7Bo0SK4uLhg69atOHPmDAwNDTF37lzExsaW+KLbxcUFT548QWhoKGrXro2wsDDs2LGjRM9RVnT+tgVmzV8GdycHeLg4Ycue/XiZnYOWTRsBAH7+awlMjAzxw/eFO2bk5eXj0bNnhf+fn4/4xGTcffgImhoasLEsui9nRSEQCPBNm07YGvo3LK1sYGZhgU3rVsPQyBh1fBpI602bMAp1fHzRsnV7AEDrdt9hwdxgOLm4w8XVHXt2bUVO9ks0aSa7t3n0i2e4deMqJk77rci5315Yp6UVRjNtbCt9Uft0q2hrQdv59T7JWg420KvujtykVGQ/Ld0bdcqCzq0D8MuCFXB3coCniyP+2XMQ2Tk5aNWk8A3fzPnLYWJkiME9OgEofC0/fFa49V9efgHiE5Nx5+FjaGloSCPbS9ZvgU+NajA3NULWy2wcPPkfLt+MxNzJo0tnkp9Jj68bYeqKUHg42KCKox02HjyJlzm5+LZhbQDAlGWbYGqoj+HftQQAhOw5iqXbD+CXwd1haWKIhJTCbRq1NETQ0hAhL78A4xb8jcjHzzAvsB8KxGJpHX0dLahV4J2gujf3xdSVW+Bpb43KjrbYePB04bVsUPjmcPKKf2BmoIfhnb4GAISEHceSnYcwa1AXWJkYIiG1cJtGLZE6tDREyMrOwbJdR9DUuwpM9HXxNC4Rf/2zD7ZmRvCp4lrsOCqULyFvvQRV3FeXEqiqqmLYsGGYPXs2Ll++jAcPHiAgIABaWloYOHAg2rZti9RU+XtKf6xvv/0Wo0aNwrBhw5CTk4NWrVph8uTJ0u0EK5KmDXyQkpaOVaFbkZScCmeHSpgzZZz0I+nY+ESZ3TISkpPRN/D1JwChu8IQuisMXpU9sODnSUX6r0jaduyK7OyXWLpgDjIzM+DuWRWTZ/4OdfXXn2DERL9Aetrrn8f6DZsgNTUFoetXIyU5CQ6Ozpg04/ci6SVHD+2FsYkpqtes/dnmU97o16oCnyPrpI8950wAADz9ezuu9SuZb0wrz/zr10VKajpWhu5AUkoqXBzs8Mek0a9fywlFX8t9xrz+DoRNu/dj0+79qFHZDQtnFF7PlNQ0zFywHInJqdDW0oRzJVvMnTwadd7Y8agiav6VF5LTM7B0+wEkpqbD1c4KC37qD2P9wje5MYnJMtdy69GzyMsvwNgFf8v0M7BtMwxqH4D45FQcv1y4Y1HXSXNl6iwL+gHeHhU3JSKgbjUkp2dgyc7DSExNh5udJRYG9nnjWqZA+Ma13HLsP+TlF+CnRRtk+hnYpil+aOsPoVCIu09jsOf0JaRnZcPUQBdfVXHBkHbNoF7M1pcVDtNLFCKQSCSS0h4Elb64W/L3KSXFxam//yNI+jCPPRqV9hAqhLqXQ0p7CBWGRmZCaQ+hwhAUVNzUoM9Ju177Ujt3dthSpfWt0eoHpfVdWvi5ABERERGRknHRTURERESkZF9I0hERERERlagy9uU4ZR2vFhERERGRkjHSTURERESK45aBCuHVIiIiIiJSMka6iYiIiEhx3KdbIYx0ExEREREpGRfdRERERERKxvQSIiIiIlIctwxUCK8WEREREZGSMdJNRERERIrjjZQKYaSbiIiIiEjJGOkmIiIiIsXxy3EUwqtFRERERKRkXHQTERERESkZ00uIiIiISGES3kipEEa6iYiIiIiUjJFuIiIiIlIcvxxHIbxaRERERERKxkg3ERERESmOkW6F8GoRERERESkZF91ERERERErGRTcRERERkZIxp5uIiIiIFMZ9uhXDRTcRERERKY43UiqEV4uIiIiISMkY6SYiIiIixTG9RCGMdBMRERERKRkX3URERERESsZFNxERERGRkjGnm4iIiIgUJ2TsVhG8WkRERERESsZINwEA1HPSSnsIFYaRSlxpD6HCsLgcUtpDqBDO1ehd2kOoMLQvXyntIVQYVprxpT2ECsG1FM/NL8dRDBfdRERERKQ4fjmOQni1iIiIiIiUjItuIiIiIiIl46KbiIiIiEjJmNNNRERERAqTMKdbIbxaRERERERKxkg3ERERESmOWwYqhJFuIiIiIiIlY6SbiIiIiBTGnG7F8GoRERERESkZF91EREREVO4tWrQI9vb20NDQQN26dREREVFs3RUrVsDX1xeGhoYwNDSEv7//O+uXBC66iYiIiEhxAoHyDgVt3rwZgYGBmDp1Ki5duoTq1asjICAAcXFxcuuHh4eja9euOHbsGM6ePQtbW1s0b94cz58//9SrUiyBRCKRKK13KjdSLh8t7SFUGFlaJqU9hApDPS+ztIdQIZyr0bu0h1BhaF++UtpDqDCsNONLewgVgquTXamdO/38XqX1rVu7pUL169ati9q1a2PhwoUAALFYDFtbWwwfPhzjx49/b/uCggIYGhpi4cKF6Nmz50eN+X14IyURERERKU6JN1Lm5OQgJydHpkwkEkEkEhWpm5ubi4sXLyIoKEhaJhQK4e/vj7Nnz37Q+bKyspCXlwcjI6NPG/g7ML2EiIiIiMqU4OBg6OvryxzBwcFy6yYkJKCgoADm5uYy5ebm5oiJifmg840bNw5WVlbw9/f/5LEXh5FuIiIiIlKYRIlfjhMUFITAwECZMnlR7pLw66+/IjQ0FOHh4dDQ0FDKOQAuuomIiIiojCkulUQeExMTqKioIDY2VqY8NjYWFhYW72w7Z84c/Prrrzh8+DCqVav20eP9EEwvISIiIqJyS11dHbVq1cKRI0ekZWKxGEeOHIGPj0+x7WbPno2ZM2di//798Pb2Vvo4GekmIiIionItMDAQvXr1gre3N+rUqYN58+YhMzMTffr0AQD07NkT1tbW0rzw3377DVOmTMHGjRthb28vzf3W0dGBjo6OUsbIRTcRERERKa4MfQ18586dER8fjylTpiAmJgZeXl7Yv3+/9ObKJ0+eQCh8Pd4lS5YgNzcXHTt2lOln6tSpmDZtmlLGyH26CQD36S5J3Ke75HCf7pLBfbpLDvfpLjncp7tklOY+3amXDiutb/2ayttFpLSUnbcoREREREQVFNNLiIiIiEhhkjKUXlIe8GoRERERESkZF91ERERERErGRTcRERERkZIxp5uIiIiIFMecboXwahERERERKRkj3URERESkMIlAUNpDKFe46CYiIiIihXHLQMXwahERERERKZlSF93h4eEQCARISUn5pH569+6Ntm3blsiYvgQCgQA7d+4s7WEQERER0f99cHrJ0qVL8dNPPyE5ORmqqoXNMjIyYGhoiPr16yM8PFxaNzw8HH5+foiMjER0dDT09fVLfOAxMTEIDg5GWFgYnj17Bn19fTg7O6NHjx7o1asXtLS0Pvkcjx49goODAy5fvgwvLy9peVZWFmbOnIl//vkHz58/h66uLjw9PREYGIg2bdp88nm/FFsOhGPDv4eQmJoGFzsbjO7TGZWd7eXWffD0BZZt+RdRD54gOiEJI3t2RNeWTWXqtB02EdEJSUXadmjeEGP7dlXGFErFjrD92LxjN5KSU+DkUAk/DuwLD1eXYuuHnzqL1RtCERMXDxsrCwzs1QNfedeUqfP46TMsX7seV2/cQkGBGJVsbTA9aDTMTU2ldW5GRmHVuk24fecehEIhnB3sMXv6RIhEIqXNtTRs23cYG3ftQ1JKKpzt7TCqXw94ujjKrfvgyXOsDN2OqAePEBOfiB/7dEXnbwJk6uzYfxQ7DhxFdHwCAMDB1hp9OrWBT81qSp9LeWDUwBuOo/tBv2YVaFiZ4UKHIYjdfaS0h1WmSCQS/Bu6BCcPb8fLrHQ4uXmh28AJMLeqVGybfdtX4fJ/RxDz/BHU1UVwdKuO9t+PhIW1vbTOiYNbcf7UPjx5EInsl5n48+8T0NLW+wwz+nwkEgk2rF+Lg/v3ITMzAx6elTFk6I+wsrZ5Z7uwf3dh+7YtSE5OgoODEwYNHgpXN3cAQHp6Gjau/xuXL11EfHwc9PT18ZVPffT4vje0tbWlfbRu2axIvz+Nm4CGjfxKdpJUbnzwotvPzw8ZGRm4cOECvvrqKwDAyZMnYWFhgXPnziE7OxsaGhoAgGPHjsHOzg5ubm5KGfSDBw9Qv359GBgYYNasWahatSpEIhGuX7+O5cuXw9raGt9++61Szg0AP/zwA86dO4cFCxbA09MTiYmJOHPmDBITE5V2zorm0JkL+GvdNozr3xWVnR0QuvcoRgTPxz9zp8FIv+gv/ezcXFibmaDpVzUx7++tcvtcM2s8xGKx9PH9py8w/Jf5aFq3ltLm8bkdPXkaS1atxaghA+Hh6oytu8Mwduov+HvJXzA0KPrm9sbtKMycMw8DenaDT+1aOHL8FCbPmo3lf86GQyU7AMDz6Bj8OH4yWvg3Qe+unaGlpYlHT55CXU1d2s/NyCiMm/YLunVsh+GD+kFFKMT9R48hEFasDLXDp89hQUgofhrUC54ujvhnz0EEzpyDTQt+haGcn8uc3BxYmZuiSb3amL9mk9w+TY0N8UOPTrC1NIcEwL5jpzD+t7+w5vcZcLSzVvKMyj4VbS2kXYvC05Bt8N66qLSHUyYd2BmCo3s3ovfwmTAxs8bu0MWYP3MIpv21HWrq8t/03rl5EY2/7gx758ooEBdg54YF+GvGYEz7aztEGpoAgNzcbFT2qo/KXvWxY8P8zzmlz2bb1s3Ys3snRgaOhbmFBTasC8GUyUFYvHQV1NXV5bY5eTwcK1csw9BhP8LV3QO7d27HlMlBWLp8NQwMDJGUmIjExET07T8QtnaVEBcbi8UL/0JSYiKCJk6R6WvEqDGoVau29LG2jo5S5/vZ8UZKhXzwX0w3NzdYWloWiWi3adMGDg4O+O+//2TK/fz8iqSXhISEwMDAAAcOHICHhwd0dHTw9ddfIzo6Wtq2oKAAgYGBMDAwgLGxMcaOHQuJRCIzliFDhkBVVRUXLlzAd999Bw8PDzg6OqJNmzYICwtD69atARRGqgUCAa5cuSJtm5KSAoFAIJ1HcnIyunfvDlNTU2hqasLFxQVr1qwBADg4OAAAatSoAYFAgMaNGwMAdu/ejQkTJqBly5awt7dHrVq1MHz4cPTt21d6Hnt7e8ycORNdu3aFtrY2rK2tsWiR7B+UlJQU9O/fH6amptDT00OTJk1w9epVmTq7du1CzZo1oaGhAUdHR0yfPh35+fnS5+/evYuGDRtCQ0MDnp6eOHTo0Pv+KcuETWFH0KZJfbRuXA+ONpYY378rNNTV8W/4Wbn1PZ3s8WOPDmherzbUVeW/VzTU04Wxgb70OHXpOmzMTVHTs/gocHmzZdcetGreFC38/WBvZ4vAIQOhIVLHvsNH5dbf9m8Y6tT0Qpf2bVDJ1gZ9e3SBi6MjdoTtl9ZZtX4T6taqgR/6fA8XJwdYW1qgft3aMov4RSvXov03LdGtYzs42NnCzsYafg3qQV1NTelz/pw2/3sArf0boVUTXzjYWuOnQb0gEqljz5ETcut7ODtiWK8u8G/wFdTU5P9cNqhdA/VqVYetlQXsrCwwqHtHaGpo4Oade8qcSrkRf+AE7kydh9hdh0t7KGWSRCLBkT0b0LLjAHjV8YONvSv6DJ+JlOR4XIk4Vmy7EZMXo16TNrCyc4atvRt6D5uBpIRoPL5/S1rH/5se+Lp9Xzi4Vv0cU/nsJBIJdu/cge+6dMdXPvXg4OCIUaPHISkxEf+dPV1su507tiHg6xbwb/417OwqYciwERCJRDh08AAAoJK9AyZMmoo6dX1gaWmF6l418H2vPog49x8KCgpk+tLW1oGhkZH0KG6hT18GhcJUfn5+OHbs9Yv82LFjaNy4MRo1aiQtf/nyJc6dOwc/P/kfn2RlZWHOnDlYt24dTpw4gSdPnmDMmDHS5//44w+EhIRg9erVOHXqFJKSkrBjxw7p84mJiTh48CCGDh0q8zHOmwQKvPOaPHkybt26hX379uH27dtYsmQJTExMAAAREREAgMOHDyM6Ohrbt28HAFhYWGDv3r1IT09/Z9+///47qlevjsuXL2P8+PEYMWKEzKK4U6dOiIuLw759+3Dx4kXUrFkTTZs2RVJSYYrEyZMn0bNnT4wYMQK3bt3CsmXLEBISgl9++QUAIBaL0b59e6irq+PcuXNYunQpxo0b98FzLy15+fmIfPgEdaq6S8uEQiFqV3XH9TsPSuwc+09FoHVjH4V+HsqyvLw83Ln3ALW8XqclCIVC1KxeDTcj78htcyvyDmpVl01jqF2zurS+WCzGfxcuwcbKCj9N/Rntvu+HwWOCcOq/CGn95JRU3L5zFwYG+hg2diLaf98fI4Km4Pqt20qYZenJy8tH1P1HqF3NU1omFArhXa0ybty5XyLnKCgQ4/Cp/5CdnYMqbs4l0idVbAmxz5GWkgCPanWlZZraunBwqYoHUVff0VLWy6wMAIC2bsmne5ZVsTExSE5OgpdXDWmZtrY2XN3cEXn7ltw2eXl5uHfvDqp7vU7BEwqF8PKqiahI+W0AIDMzE1paWlBRUZEpX7pkAbp16YDAkcNw6OD+IkHE8k4iECrtqIgU2jLQz88PI0eORH5+Pl6+fInLly+jUaNGyMvLw9KlSwEAZ8+eRU5ODvz8/PDgQdEF1Ku6Tk5OAIBhw4ZhxowZ0ufnzZuHoKAgtG/fHkBhLvmBAwekz9+7dw8SiaRI6oqJiQmys7MBAEOHDsVvv/32QXN68uQJatSoAW9vbwCFEepXTP+fz2psbAwLCwtp+fLly9G9e3cYGxujevXqaNCgATp27Ij69evL9F2/fn2MHz8eAODq6orTp0/jzz//RLNmzXDq1ClEREQgLi5OmhM7Z84c7Ny5E1u3bsXAgQMxffp0jB8/Hr169QIAODo6YubMmRg7diymTp2Kw4cPIzIyEgcOHICVlRUAYNasWWjRosU755yTk4OcnBzZstxciD7TO/CUtAwUiMVF0kiM9PXw+HlsiZzj+PmryMh8iVaNfEqkv7IgNS0dYrG4SBqJoYE+njx/LrdNUkqKnPoGSE5OAQCkpKbi5ctsbNq2E317dMGgXt0RcekKpgTPwdxfpsKrSmVExxT+m6zd9A9+6NMTzg72OHjsOEZPmoHVC+fCxsqy5CdbClLS0wt/Lt+6Xkb6enjyPLqYVh/m/uOnGDThZ+Tm5kFTQ4RZY4fDwZapJfR+aSmF9wLoGRjLlOvpGyE15cNSGsViMf5Z8zuc3L1gbfflvNlLTi4MYBkYGsqUGxgYIjk5WW6btLTUwt+zcto8e/pUbpvU1FRs3rQBAS1aypR379EL1ap7QaShgcuXLmDJovl4+fIlvm3T7mOnROWcQm8lGjdujMzMTJw/fx4nT56Eq6srTE1N0ahRI2led3h4OBwdHWFnZye3Dy0tLemCGwAsLS0RFxcHoPAHNzo6GnXrvn5Hr6qqKl0Qv0tERASuXLmCypUrF1lQvsvgwYMRGhoKLy8vjB07FmfOnHlvm4YNG+LBgwc4cuQIOnbsiJs3b8LX1xczZ86Uqefj41Pk8e3bhdHBq1evIiMjA8bGxtDR0ZEeDx8+xP3796V1ZsyYIfP8gAEDEB0djaysLNy+fRu2trbSBbe8c8oTHBwMfX19mePP1fLzUcur3cdOw8erMkyNDEp7KGWaWFwYdalX1xud2nwDZ0cHdOvYDj61a+LffYWfyoj/H5n5JqAZWvj7wcXJAUP794attRX2HZKf1kKy7KwsETJnBpb/OgVtA5rgl4Ur8fCp/DdK9GU7dyIMP3b3kR4FBfnvb/Qem1YE48WTexgQ+GHBqPIq/NgRdGrfWnrkl8C1e5+srEzMmDoJtnaV0K17T5nnunTrAc/KVeDk5IyOnbqgfcfvsGPbFqWP6XOSQKC0oyJSKNLt7OwMGxsbHDt2DMnJyWjUqBEAwMrKCra2tjhz5gyOHTuGJk2aFNuH2ls5oAKBQKGPW5ydnSEQCBAVFSVT7uhYuLuApqamtEz4/5u83uw/Ly9Ppl2LFi3w+PFj7N27F4cOHULTpk0xdOhQzJkz553jUFNTg6+vL3x9fTFu3Dj8/PPPmDFjBsaNG/dBOVsZGRlFcuRfMTAwkNaZPn26NOr/plc3rX6MoKAgBAYGypS9vP3+NxslxUBPBypCIZJS02TKk1LTYGTw6XfOR8cn4vz1SPw6etAn91WW6OvpQigUIjklVaY8OSUVRv//mXmbkYGBnPopMDQ0kPapoqICe1tbmTp2Nja4fisSAGD8/7r2trJ3+9vZWiM2IeEjZ1P2GOjqFv5cvnW9Cn8uP+0jeTU1VdhYmgMA3J3sEXnvIbaEHcLYH3p/Ur9U8VSv3RgOLq9zrPPzcgEAaSmJ0Dd8vZtQWmoSbO1d39vfphXBuH7xBMbMXA1DY/OSH3AZUqeuj3SHEeD13/uU5GQYGb3+pCAlJRmOjk5F2gOAnp5+4e/ZtyLhKSnJMDSSjX5nZWVh6uQJ0NTSxMTJ06Q7uxXHzc0DmzdtQF5eLtTUmNv9JVI4aebVDZLh4eHSGwuBwujvvn37EBERUWw+9/vo6+vD0tIS586dk5bl5+fj4sWL0sfGxsZo1qwZFi5ciMzMzHf29yo95M0bNd+8qfLNer169cL69esxb948LF++HACki+e3b4yQx9PTE/n5+dIUFwAyN5e+euzh4QEAqFmzJmJiYqCqqgpnZ2eZ41VOec2aNREVFVXkeWdnZwiFQnh4eODp06cy83v7nPKIRCLo6enJHJ8rtQQA1FRV4e5gh/M3Xr9xEovFOH8jClVd5W/Npog94WdhqK+L+jWqfHJfZYmamhpcnR1x6ep1aZlYLMala9dR2V3+H19Pd1dcunZdpuzilWvS+mpqanB3ccLTt9JTnr14AXOzwp9DC3MzmBgZ4unzF7J1nkfLbClY3qmpqcLNyR4Xrr/O2xSLxbh47RaquMr/A/2xxBIJct8KABABgIamNsws7aSHpa0T9AxMEHn99X0WL7My8PDudTi6VS+2H4lEgk0rgnEl4ihGTVsOE/OKn86kpaUFKytr6WFnVwmGhka4evWytE5WVibuREXC3cNTbh9qampwdnbFtTfaiMViXL1yGW7ur9tkZWViyqTxUFVVxaQpMz4o2PbgwT3o6Ohywf0FU/hr4P38/DB06FDk5eVJI90A0KhRIwwbNgy5ubkfvegGgBEjRuDXX3+Fi4sL3N3dMXfu3CJfrrN48WLUr18f3t7emDZtGqpVqwahUIjz588jMjIStWoVbhGnqamJr776Cr/++iscHBwQFxeHSZMmyfQ1ZcoU1KpVS5qWsmfPHunC2MzMDJqamti/fz9sbGygoaEBfX19NG7cGF27doW3tzeMjY1x69YtTJgwAX5+ftDTex2pPX36NGbPno22bdvi0KFD2LJlC8LCwgAA/v7+8PHxQdu2bTF79my4urrixYsXCAsLQ7t27eDt7Y0pU6bgm2++gZ2dHTp27AihUIirV6/ixo0b+Pnnn+Hv7w9XV1f06tULv//+O9LS0jBx4sSPvvafU9dWTTFjyVp4ONrB09keoXuPIjsnB9/8Pwd72qIQmBoZYGjXtgAKb4x8+KzwzUVeQQHik1Jw59FTaGqIYGthJu1XLBZjz/GzaNXwK6i+dUNLRdCpzTf4dd4iuDo7SbcMzM7OwddNC19zs/5cAFMjIwzo1R0A0KF1K4ycMBX/7PgXX9WuiaMnTiPq3n2MHvr6U4DO7b7FjN//RLXKnqhRtTIiLl3BmYiLmDdrGoDCT6M6t2uDkE2b4eRQCc4O9jhw9DiePH+OaeNHf/ZroEydWwfglwUr4O7kIN0yMDsnB62a+AIAZs5fDhMjQwzu0QlA4c2XD58VvmHJyy9AfGIy7jx8DC0NDWlke8n6LfCpUQ3mpkbIepmNgyf/w+WbkZg7uWJdu4+loq0FbefX6YhaDjbQq+6O3KRUZD/9tFz6ikAgEKDpN92xd+sKmFnawcTMGrs2LYKBoSm86rz+Wzt32kDUqNMEfi27AAA2rZiFiJP7MGT8PGhoaiM1ufBTKU0tHaiLCj8pTU1OQFpKAuJjCnOVnz++Bw1NLRiZWFaIGy4FAgG+bdsOm0M3wsrKGubmlli/LgRGxsb4yuf1PVgTg36CT736+KZ1WwBA23Yd8Ofc2XB2cYWrqxt27dqB7Jxs+Dcr3IM/KysTUyaOR05ODkb/NB4vs7LwMisLAKCnrw8VFRVEnDuL5ORkuLt7QE1dHVcuX8KWzaFo16HjZ78OylRRb3hUlo9adL98+RLu7u4wN3/9UVWjRo2Qnp4u3VrwY40ePRrR0dHo1asXhEIh+vbti3bt2iE19fVHvk5OTrh8+TJmzZqFoKAgPHv2DCKRCJ6enhgzZgyGDBkirbt69Wr069cPtWrVgpubG2bPno3mzZtLn1dXV0dQUBAePXoETU1N+Pr6IjQ0FEBhPvn8+fMxY8YMTJkyBb6+vggPD0dAQADWrl2LCRMmICsrC1ZWVvjmm28wZYrs/pyjR4/GhQsXMH36dOjp6WHu3LkICCh80QoEAuzduxcTJ05Enz59EB8fDwsLCzRs2FB6XQMCArBnzx7MmDEDv/32W2FU0t0d/fv3B1CYPrNjxw7069cPderUgb29PebPn4+vv/76o6//59KsnjdS0jKwfMseJKakwbWSDeaNHw7j/6eXxCYkQfjGriPxSan4fvws6eMNew5jw57DqOnhgiVTX6fKRFyPRExCElo3rvf5JvMZNfGtj9TUNIRs3Fz45TiO9vht2kQY/T8FJC4+Qea6VfFww6TRI7B6wyasXLcR1laWmDlhrHSPbgDw9amLUYMHYuPWHViwYjVsra0wffwYVPX0kNbp2KYVcvNysWjVWqSnZ8DJoRLmzJgMa8vXNxhXBP716yIlNR0rQ3cgKSUVLg52+GPSaGl6SWxCosxuOAnJyegzZqr08abd+7Fp937UqOyGhTOCAAApqWmYuWA5EpNToa2lCedKtpg7eTTqVK9Yn8R8LP1aVeBzZJ30seecCQCAp39vx7V+QaU1rDIloG1v5Ga/xPqlM5GVmQ5n9xr4cfJimT26E2KeIiP9dUrE8QOFucN/TOkv01evodNRr0nhl7idOLgFe/5ZJn1uzuS+ReqUdx06dkZ2djYWLpiHzIwMeFaugukzgmUi0zHR0Uh7I93Rt1FjpKalYMO6tUhOLkxFmT5jlvTmyvv37iEqqjD9bmC/XjLnW7lmHczNLaCiooq9e3Zj1YqlkEgksLSyQr8BgxDwtezNlvRlEUgq2v41ZYS9vT1GjhyJkSNHlvZQPkjKZd4QV1KytExKewgVhnreu1PI6MOcq9G7tIdQYWhfvlLaQ6gwrDTjS3sIFYKrk/yNKz6H+FsR76/0kUw96yit79LCzwWIiIiIiJRM4fQSIiIiIiIJY7cK4aJbSR49elTaQyAiIiKiMoJvUYiIiIiIlIyRbiIiIiJSmERQMb85UlkY6SYiIiIiUjJGuomIiIhIYfxyHMXwahERERERKRkj3URERESkMAmY060IRrqJiIiIiJSMi24iIiIiIiXjopuIiIiISMmY001ERERECuPuJYrhopuIiIiIFMYvx1EM36IQERERESkZI91EREREpDBuGagYRrqJiIiIiJSMi24iIiIiIiXjopuIiIiISMmY001ERERECuOWgYrh1SIiIiIiUjJGuomIiIhIYdy9RDFcdBMRERGRwpheohheLSIiIiIiJeOim4iIiIhIybjoJiIiIiJSMuZ0ExEREZHCeCOlYhjpJiIiIiJSMka6iYiIiEhh3L1EMbxaRERERFTuLVq0CPb29tDQ0EDdunURERHxzvpbtmyBu7s7NDQ0ULVqVezdu1ep4+Oim4iIiIgUJoFAaYeiNm/ejMDAQEydOhWXLl1C9erVERAQgLi4OLn1z5w5g65du6Jfv364fPky2rZti7Zt2+LGjRufelmKJZBIJBKl9U7lRsrlo6U9hAojS8uktIdQYajnZZb2ECqEczV6l/YQKgzty1dKewgVhpVmfGkPoUJwdbIrtXM/uH9faX07OjkpVL9u3bqoXbs2Fi5cCAAQi8WwtbXF8OHDMX78+CL1O3fujMzMTOzZs0da9tVXX8HLywtLly79tMEXg5FuIiIiIipTcnJykJaWJnPk5OTIrZubm4uLFy/C399fWiYUCuHv74+zZ8/KbXP27FmZ+gAQEBBQbP2SwBspCQBwQaV+aQ+hwrgWqV7aQ6gwBlrseX8lei9GZ0tOZg2v0h5ChaEfWKe0h1Ax/Lau1E4tEShvy8Dg4GBMnz5dpmzq1KmYNm1akboJCQkoKCiAubm5TLm5uTkiIyPl9h8TEyO3fkxMzKcN/B246CYiIiKiMiUoKAiBgYEyZSKRqJRGUzK46CYiIiIihUkkyot0i0SiD15km5iYQEVFBbGxsTLlsbGxsLCwkNvGwsJCofolgTndRERERFRuqauro1atWjhy5Ii0TCwW48iRI/Dx8ZHbxsfHR6Y+ABw6dKjY+iWBkW4iIiIiUpikDMVuAwMD0atXL3h7e6NOnTqYN28eMjMz0adPHwBAz549YW1tjeDgYADAiBEj0KhRI/zxxx9o1aoVQkNDceHCBSxfvlxpY+Sim4iIiIjKtc6dOyM+Ph5TpkxBTEwMvLy8sH//funNkk+ePIFQ+PpNQr169bBx40ZMmjQJEyZMgIuLC3bu3IkqVaoobYzcp5sAAIevyd+GhxR37SF3Lykp3L2kZFzQ9n9/Jfog3L2k5Hhz95ISYV6Ku5fcvf9YaX27OFVSWt+lhZFuIiIiIlLYx3xz5Jes7CTjEBERERFVUIx0ExEREZHCGOlWDCPdRERERERKxkg3ERERESmMkW7FMNJNRERERKRkXHQTERERESkZF91ERERERErGnG4iIiIiUhhzuhXDRTcRERERKUwi4aJbEUwvISIiIiJSMka6iYiIiEhhTC9RDCPdRERERERKxkU3EREREZGScdFNRERERKRkzOkmIiIiIoUxp1sxjHQTERERESkZI91EREREpDBGuhXDSDcRERERkZIx0k1ERERECuM3UiqGkW4iIiIiIiXjopuIiIiISMmYXkJEREREChPzRkqFMNJNRERERKRkjHQTERERkcK4ZaBivqhI96NHjyAQCHDlypVi64SHh0MgECAlJQUAEBISAgMDg88yPiIiIiKqmD57pLt3795ISUnBzp07ZcrDw8Ph5+eH5ORkpS1ybW1tER0dDRMTkw9u07lzZ7Rs2VL6eNq0adi5c+c7F+7ypKenY/LkydixYwfi4uJQo0YN/PXXX6hdu7ZC/VQ0EokEYZsX4/SRbXiZmQ5Hdy90GTAJZpaVim1zYMdKXDl3BLHPH0JNXQRHNy+07T4S5tYO0jobl81A1PX/kJoUD5GGFhzcqqNtj1GweKNORSeRSHDh4AJERmxBzss0WNjXhG+7qdA3tS+2zYWDC3Dx8CKZMgNTB3T+aZ+SR1t2/HP4NP7eG47E1HS42Fpi7PftUMXJTm7d7cf+Q9jpi7j/LAYA4GFvg6GdWkjr5+UXYMm2fTh1NRLP4xKho6WJupVdMPy7ljA11P9scyoNEokE/4YuwcnD2/EyKx1Obl7oNnACzK2Kf23v274Kl/87gpjnj6CuLoKjW3W0/34kLKztpXVOHNyK86f24cmDSGS/zMSff5+AlrbeZ5hR2WbUwBuOo/tBv2YVaFiZ4UKHIYjdfaS0h1WmaPr4Q7thSwh19ZEf/RRpu/5G/rMHxdYXaGhBJ6ATRFW8IdTSRkFyAtL/3YDcqKvSOkI9Q+i06AyRWzUI1EXIT4hF2pYVyH/+8HNMqdRxy0DFfFGRbhUVFVhYWEBV9cPfa2hqasLMzOyTz92/f38cOnQI69atw/Xr19G8eXP4+/vj+fPnn9x3eXZo1xqE79uILgMn46fgDVAXaWLhzz8gLzen2DZ3b15Aw4AuGDNrPYZPXo6C/Hws+PkH5GRnSevYOXqix5AZmDxvJ4ZOWgJIJFg4cxDEBQWfY1plwtXwlbhxeh18209Du+H/QFVdE2Gr+iM/r/hrCwCG5i74fvJJ6fHtkI2fZ8BlwMH/rmDuxt0Y2LYZNswYCVc7Kwz7fQWS0tLl1r8YeR8BX3lhWdAPWDNlOMyN9TH09+WIS0oFAGTn5iLy0XP0b+OPDTNHYc6PvfAoOg6j/lzzOadVKg7sDMHRvRvRfdBEjA9eB5GGJubPHPLO1/admxfR+OvOGB/8N0ZMXYqCgnz8NWMwcrJfSuvk5majsld9tGjf73NMo9xQ0dZC2rUo3PhxemkPpUwSVasL3W+6IePIDiTOn4y86Ccw7DcWguLesKmowLD/OKgYmiB1/XwkzBmLtG2rIU5LllYRaGrBaPBkQFyA5NVzkPDHeGSEbYTkZeZnmhWVN2Vy0T1t2jR4eXnJlM2bNw/29vbSx71790bbtm0xa9YsmJubw8DAADNmzEB+fj5++uknGBkZwcbGBmvWvP7jJi+9ZO/evXB1dYWmpib8/Pzw6NEjmfO+mV4SEhKC6dOn4+rVqxAIBBAIBAgJCUHfvn3xzTffyLTLy8uDmZkZVq1ahZcvX2Lbtm2YPXs2GjZsCGdnZ0ybNg3Ozs5YsmSJtM26devg7e0NXV1dWFhYoFu3boiLi5M+/yr15cCBA6hRowY0NTXRpEkTxMXFYd++ffDw8ICenh66deuGrKwslHUSiQTHwtbj6w4DUL22H6wruaLXsF+QmhyPq+ePFttu2KSl8PFrAytbZ9jYu+H7oTORnBCNJw9uSes0aNYRLp7eMDazhp2jJ1p3HY7kxBgkxr/4HFMrdRKJBNdP/Y2aTX+AfeWmMLZ0g1/n35CVFodHNw+/s61QqAItXVPpoalt+JlGXfrW7z+Odo3r4tuGdeBobYEJvTtAQ6SGXcfPy63/y+Du+M6/PtwqWcPBygyT+30HiViCiFt3AQC6WppYPG4Qmtf1gr2lGao6V8K4nu1w+9EzRCcky+2zIpBIJDiyZwNadhwArzp+sLF3RZ/hM5GSHI8rEceKbTdi8mLUa9IGVnbOsLV3Q+9hM5CUEI3H91+/tv2/6YGv2/eFg2vVzzGVciP+wAncmToPsbve/fr+Umn7tsDLiHBkXziJgrgXSN+xBpK8HGjWbii3vqZ3Iwi0tJHy9zzkPb4LcXIC8h5GIj/6yes+G32DgtSkwsj2swcQJ8cj9+4NFCTFye2TqFzfSHn06FHY2NjgxIkTOH36NPr164czZ86gYcOGOHfuHDZv3oxBgwahWbNmsLGxKdL+6dOnaN++PYYOHYqBAwfiwoULGD16dLHn69y5M27cuIH9+/fj8OHCX2z6+vpwdXVFw4YNER0dDUtLSwDAnj17kJWVhc6dOyM/Px8FBQXQ0NCQ6U9TUxOnTp2SPs7Ly8PMmTPh5uaGuLg4BAYGonfv3ti7d69Mu2nTpmHhwoXQ0tLCd999h++++w4ikQgbN25ERkYG2rVrhwULFmDcuHEffW0/h8S450hLSYBb1a+kZZraurB3roqHUVfhXb/FB/XzMisDAKCtI//j+pzsLJw9thPGZtYwNLb49IGXA+lJz5CVHg9rl3rSMpGmLsxsqyH28RU4e7Uqtm1qwmOsm+kLFTURzO28UKdFIHQNrT7HsEtVXn4+Ih89R5/WTaVlQqEQdTxdcP3e4w/qIzsnF/kFBdDT1iq2TkZWNgQCAXS1NT95zGVVQmzha9ujWl1pmaa2LhxcquJB1FXUbvD1B/UjfW3rVuxUHFIyFRWoWtsj89i/r8skEuTeuwk1O2e5TUSeNZH3+B502/aCyLMmJJnpeHnlDLLC9wASibROzp3r0O8+HOqO7ihITcLL/47gZUT4Z5hU2cAbKRVTKovuPXv2QEdHR6as4CM+9jcyMsL8+fMhFArh5uaG2bNnIysrCxMmTAAABAUF4ddff8WpU6fQpUuXIu2XLFkCJycn/PHHHwAANzc3XL9+Hb/99pvc82lqakJHRweqqqqwsHi9eKtXrx7c3Nywbt06jB07FgCwZs0adOrUSTpPHx8fzJw5Ex4eHjA3N8emTZtw9uxZODu/fsH37dtX+v+Ojo6YP38+ateujYyMDJnr9fPPP6N+/foAgH79+iEoKAj379+Ho6MjAKBjx444duxYsYvunJwc5OTIfsSbmwuoq4vk1leWtJQEAICegbFMua6BMdJSEj+oD7FYjG0hs+HoVgNWdi4yz504EIod6/5Ebs5LmFvZY/jk5VBVUyuZwZdxWenxAABNHdlrq6lrgqz0hGLbmdlVR+POwTAwdUBWWhwuHl6E3Ut6oFPgbqhr6BTbriJISc9EgVgMYz3ZeRrr6+JR9IdFruZvDoOJoT7qVnaR+3xObh7m/xOGgK+8oKOpIbdORVDca1tP3wipCry2/1nzO5zcvWBdzMKI6EMItXQhUFGBOCNVplycngZ1U/kBBRUjU6g7eSD7ylmkrJkDFWNz6LXtBYGKKjIP75DW0fqqCbJO7kfysd1Qs3GE7rffQ5Kfj+xLp+T2S1+2Ukkv8fPzw5UrV2SOlStXKtxP5cqVIRS+noK5uTmqVn39kaOKigqMjY1lUjTedPv2bdStW1emzMfHR+FxAIU5269SWWJjY7Fv3z6ZRfS6desgkUhgbW0NkUiE+fPno2vXrjLjv3jxIlq3bg07Ozvo6uqiUaNGAIAnT57InKtatWoyc9bS0pIuuF+VFTdnAAgODoa+vr7MEbpq9kfNWxERJ8Mwqkdd6VGQn//JfW5e+QtePL2HvqOKvlGq3aAVgn7/ByOnr4aZZSWsmjvmnfmk5dndS/9i1aSa0kNc8HHX1s69IZyqfQ1jSzfYuvmiRd/lyM1Ow4Nr+0t4xBXPmn+P4uC5K/jjx14QqRd9c5eXX4Dxi9ZBIgGCencohREqz7kTYfixu4/0KPjIn783bVoRjBdP7mFAoPwgCJFSCQQQZ6Yhbdsq5D9/hJxr55B5bDc06zZ5o44QeS8eI+PAFuS/eIyXEcfwMiIcml81Kb7fCkYiESjtqIhKJdKtra0tE+EFgGfPnkn/XygUQvL/j29eycvLK9KP2ltRS4FAILdMLBZ/6pDfq2fPnhg/fjzOnj2LM2fOwMHBAb6+vtLnnZyccPz4cWRmZiItLQ2Wlpbo3LmzdLGcmZmJgIAABAQEYMOGDTA1NcWTJ08QEBCA3NxcmXO9OcePmXNQUBACAwNlyk7d+eipf7Bq3o1h7/z6TVF+fuG80lISoW9oKi1PT0mEjb3be/vbvHIWblw6gVHT18hNG9HU1oWmti7MLCvBwaU6fupTH1cjjsC7QUs5vZVvlTz90NHu9Zuxgv9f25cZidDWe30j8Mv0BBhbeXxwvyJNPeib2CM18cPSK8ozA11tqAiFSEzLkClPTE2Hif67d8f4e284QsKOYsnYQXCxKxo5e7Xgjk5IxtLxP1S4KHf12o3h4PLGaztP/ms7LTUJtvau7+1v04pgXL94AmNmroahsXnJD5i+KOKsdEgKCiB8KwVRqKuHgvQU+W3SUyEpyJemkgBAftwLqOgZACoqQEEBxOkpKIiV3QwhP+4FRFW8S3oKVEGUyRspTU1NERMTI7PwVnSLvg/h4eGBiIgImbL//vvvnW3U1dXlpsIYGxujbdu2WLNmDUJCQtCnTx+57bW1tWFpaYnk5GQcOHAAbdq0AQBERkYiMTERv/76K3x9feHu7v7OaPWnEIlE0NPTkzk+R2qJhqY2zCztpIeljRP0DEwQdeOctM7LrAw8uncdDm7Vi+1HIpFg88pZuBpxFCOmroSJedF8/SJtIIFEIv/NW0WgrqEDfZNK0sPQ3BlauqZ4fvestE5udgbinl6DeSWvD+43LycTaYlPoaVr+v7K5Zyaqirc7a1x/uZdaZlYLMb5W/dQ1bn4be7Whh3Dyl2HsXDMAHg62hZ5/tWC+2lMPJaMGwQDXW2ljL80FXlt2xa+tiOvv/79+jIrAw/vXofje17bm1YE40rEUYyathwm5tafY/hU0RUUIP/5I6g7e74uEwig7lwZeU/uyW2S9+gOVI3NAcHriKuKiQUK0pKB/68Bch/dgYqppUw7FRMLFHxgClVFIIFAaUdFVCYX3Y0bN0Z8fDxmz56N+/fvY9GiRdi3r+T3Cf7hhx9w9+5d/PTTT4iKisLGjRsREhLyzjb29vZ4+PAhrly5goSEBJnc6P79+2Pt2rW4ffs2evXqJdPuwIED2L9/Px4+fIhDhw7Bz88P7u7u0sW5nZ0d1NXVsWDBAjx48AC7d+/GzJkzS3zOZYlAIIBfqx7Yv205rp0/hueP7+DvhROhb2iK6rVffzz31/T+CN+3Sfp488pfcP5kGPqM+BUiDW2kJicgNTkBuTnZAICE2Gc4sGMlnty/haT4aDyIuoJVf4yGuroIVWo2+OzzLA0CgQBVG/TEpaNL8ejmUSRGR+HY5nHQ0jODfWV/ab1/l/fGjdPrpY/P7vkNL+5HID3pGWIeXcKBv4dDIBTC2esbeaepcHp83Qg7jp/DvyfP4+HzWASv3Y6XObn4tmHhfvpTlm3Cgn9e39gcsucolmzbj6n9v4OliSESUtKQkJKGrOzC3wt5+QUYt+Bv3H74FD8P7o4CsVhaJ68E0qvKKoFAgKbfdMferStw9Xw4nj++izXzJ8HA0BRedfyk9eZOG4hje0OljzetmIVzJ8LQb2QwNDSLvrYBIDU5AU8fRiI+5ikA4Pnje3j6MBKZ6bL5ul8aFW0t6FV3h151dwCAloMN9Kq7Q8PW8j0tvwyZJ/dBs05jaNRsABUzK+i26w2BmgjZF04AAPS+GwSdr7+T1s/67wgEWjrQbd0DKiYWUHevDm2/b/HyzOvdYbJO7YeanRO0/FpDxdgMGl4+0KrrJ1OH6E1lcvcSDw8PLF68GLNmzcLMmTPRoUMHjBkzBsuXLy/R89jZ2WHbtm0YNWoUFixYgDp16mDWrFkyudhv69ChA7Zv3w4/Pz+kpKRgzZo16N27NwDA398flpaW/2vvvuNrvv4/gL/uzZK9ZBAii9gSlMYO2hIrhFJRkiYtVaP2qERRtL6xqVEkRtWvRhRFqKBixYp0SBAjRoZmD4kk9/7+SN26EuSSm8/9xOv5eNxHcz/rvu+p3Lzv+bzPOWjSpAlq11a+xZyVlYUZM2bg/v37sLCwgI+PD+bPn68oDbGyskJYWBhmzpyJFStWoGXLlggJCUHfvn0r9T1rmvf6+eNJwWNsXze3dAGNhu744qs10Hmm5/2flPvIy/lverVTR34GACz7Wvn/07DR8+Dh2Q/aOrq4ee0yjv+6Dfm52TA2s4RLo1aY9M0WGJsqD+yqzlp0CUTRk8f4fXcwnhRkw9ahFbwCfoC2zn9tm52WiIK8/9o2LysFx7ZPQkF+JvSNLGDr0AreY/4P+kYWQryFKvf+u27IyMnF2j0RSMvKQQP72lg5JRCWpsYAgOS0DEie6fnaFXkWRcUlmLpyi9J1PvN+DyMHfIBHGVk4eeUvAMBHs5YoHbNuxii0blR9Bwh+4O2HJwWPsW3tPOTn5cCloTvGBX2v/LudfA+5z/xun4zYCQBYHByodK0RX8xBu66ldwV/P7ITB35ep9gXEvRJmWPeRqatmsLj2FbF88YhpRMK3NuyB7EBM4QKS2MUxp5HjqExjN73KV0c52EiMjb9D7LcbACAlpmlUimJLCsdmRsXwaiPLyy/nI+S7Azkn44onb3kX8X3byNzy3IY9fgQRt28UZLxCDn7t6Eg5kyVvz8SB4n8+eJpem25ubmws7NDaGgoBgwYIHQ4KvkttnoOMBRC7G1doUOoNj6zPfDqg+iVLhp2f/VBVCF57m5Ch1BttJ7YRugQqgWb77a++iA1uRCfqbZrv+NqprZrC0Uje7rFRiaT4Z9//sHixYthZmZW7XuniYiIiKrrLCPqwqS7EiQmJsLR0RF16tRBWFiYSsvMExEREYmR+ueGq16YHVYCBweHMlMcEhERERE9xaSbiIiIiFTG8hLVaOSUgURERERE1QmTbiIiIiIiNWPSTURERESkZqzpJiIiIiKVVdfl2tWFPd1ERERERGrGnm4iIiIiUhlnL1ENk24iIiIiUhnLS1TD8hIiIiIiIjVj0k1EREREpGZMuomIiIiI1IxJNxERERGpTCZX30Od0tPT4evrCxMTE5iZmSEgIAC5ubkvPX7s2LFwdXWFvr4+7O3tMW7cOGRlZan0uky6iYiIiOit4evri7/++gtHjx7FgQMH8Pvvv+Ozzz574fEPHz7Ew4cPERISgj///BNhYWE4fPgwAgICVHpdzl5CRERERCoT4+wl165dw+HDh3HhwgW0bt0aALBy5Up4eXkhJCQEtWvXLnNO06ZNsXv3bsVzZ2dnzJ8/H8OGDUNxcTG0tSuWTrOnm4iIiIg0SmFhIbKzs5UehYWFb3zds2fPwszMTJFwA0D37t0hlUpx/vz5Cl8nKysLJiYmFU64ASbdRERERPQa5HKJ2h4LFy6Eqamp0mPhwoVvHHNycjKsra2Vtmlra8PCwgLJyckVusY///yDefPmvbQkpTxMuomIiIhIo8yYMQNZWVlKjxkzZrzw+OnTp0Mikbz0ERcX98ZxZWdno1evXmjcuDG+/vprlc5lTTcRERERaRQ9PT3o6elV+PhJkybBz8/vpcc4OTnB1tYWqampStuLi4uRnp4OW1vbl56fk5ODHj16wNjYGOHh4dDR0alwfACTbiIiIiJ6DXI1T+2nCisrK1hZWb3yOA8PD2RmZuLSpUto1aoVACAyMhIymQxt27Z94XnZ2dn44IMPoKenh3379qFGjRoqx8jyEiIiIiJ6KzRq1Ag9evTAp59+iujoaJw+fRpjxozBkCFDFDOXPHjwAA0bNkR0dDSA0oT7/fffR15eHjZu3Ijs7GwkJycjOTkZJSUlFX5t9nQTERERkcpkIpwyEAB+/PFHjBkzBt26dYNUKoWPjw9WrFih2F9UVIT4+Hjk5+cDAC5fvqyY2cTFxUXpWrdv34aDg0OFXpdJNxERERG9NSwsLLB9+/YX7ndwcID8mdqZLl26KD1/XUy6iYiIiEhlcrk4e7qFwppuIiIiIiI1Y9JNRERERKRmLC8hIiIiIpVp0pSBYsCebiIiIiIiNWNPNxERERGpTC7SKQOFwp5uIiIiIiI1Y083EREREalMxppulTDpJgCAk+F9oUOoNmLhJHQI1YakpEjoEKqF2vqPhA6h2jCd2EboEKqNi0uihQ6hWuj1ndARUEWxvISIiIiISM2YdBMRERERqRnLS4iIiIhIZVwGXjVMuomIiIhIZVwcRzUsLyEiIiIiUjP2dBMRERGRymRcHEcl7OkmIiIiIlIz9nQTERERkcpY060a9nQTEREREakZk24iIiIiIjVj0k1EREREpGas6SYiIiIilXFxHNUw6SYiIiIilck4kFIlLC8hIiIiIlIz9nQTERERkco4ZaBq2NNNRERERKRmTLqJiIiIiNSMSTcRERERkZqxppuIiIiIVCYHpwxUBXu6iYiIiIjUjD3dRERERKQyztOtGvZ0ExERERGpGXu6iYiIiEhlnKdbNezpJiIiIiJSMybdRERERERqxvISIiIiIlIZy0tUw55uIiIiIiI1Y083EREREalMJufiOKpgTzcRERERkZqxp5uIiIiIVMaabtWwp5uIiIiISM3eqqT7zp07kEgkiImJeeExJ06cgEQiQWZmJgAgLCwMZmZmVRIfEREREVVPVV5e4ufnh8zMTOzdu1dp+4kTJ+Dp6YmMjAy1Jbl169ZFUlISatasWeFzBg8eDC8vL8Xzr7/+Gnv37n1p4l6eNWvWYM2aNbhz5w4AoEmTJggODkbPnj1Vuo6Y7d+/H7t270ZGRgacHB3x+eefw9XV9YXHnzp1Clu2bkVKSgrsateG/yefoM077yj2y+VybN22DYcPH0ZeXh4aN26MMV98ATs7O6XrREdHY/v27bh95w50dXXRrGlTBAcHq+19agq5XI6LR1YiLnonCh9nw9ahJTr2nw1TK4cXnnPxyEpc+m210jYzK0cMnnJIzdFqjv87dhZbDv2OtKxcNLC3xVTfvmjqVLfcY/ecjMaB01eQ8CAZANDIwQ5jfD5QOn72hp3Yf/qy0nkeTetj9aRP1PcmBCCXy/Hjts04cvgQ8vJy0ahxE4z+Yhxq29V56Xm/7v8Fe3bvREZGOhwdnTHy8y/QwLUhACAnJxvbt23BlcuX8OhRKkxMTfGuR3sM+9gPhoaGimv08XqvzHWnTJuJTp09K/dNCkTfozsMO3lBamyK4qR7yP5lC4rv33rh8ZIaBjD6YBD0mraG1MAQJRn/IGf/j3gSf1VxjNTEHEY9B0PPtTkkunoo/icF2Tt/QPGD21XxljSaRYfWcJoUANOWTVGjtjUu+oxGyr5jQoelkVheopq3qqZbS0sLtra2Kp2jr68PfX39N37tOnXq4Ntvv0X9+vUhl8uxefNm9OvXD1euXEGTJk3e+Pqa7uTJk1j/ww8YO2YMXBs2xN69ezErKAg/rF9f7pesv//+G99+9x38/fzQpk0bnDhxAvPmzcPKFSvg4OAAANi5axf27duHSRMnwtbWFlu2bsWsoCCsW7sWurq6AICoqCgsX7ECfiNGoEWLFiiRyXD33y8+1d3VExvw5+mt8Bz8LYwt6uBCxHL8ujEQH076Fdo6ei88z9ymPnp/tknxXCJ9ez4mIs7HYsmOXzFzuDeaOdXFj0dP44vFmxC+cBIsTIzKHH8p7hZ6vNscLVz6QFdHG2EHf8fokE3YNf9LWJubKo5r16wBvg4YqHiuq1392nT3rv/DgX178eXEqbCxtcWPW8MQHDQD36/dqPh9fN6pkyew4Yd1+GLMODRo2Aj79u5BcNAMrF2/CWZm5khPS0NaWho+CfwMde3rITUlBd+vWo70tDTM+Er5i/P4CZPRqtV/X8oNjcr+/xIjveZtYdx7KLLDQ1GUmACDDj1gHjAV/4RMhTwvu+wJWlowD5wGWW42sratQEl2BrTMakJekK84RKJvAIvPg/Dk1jVkbAqBLC8H2jVtIH+cV4XvTHNpGRogOzYe98J2o/Wu1a8+gaiCNLK85Ouvv4abm5vStmXLlimSLaC0x9zb2xsLFiyAjY0NzMzMMHfuXBQXF2PKlCmwsLBAnTp1EBoaqjinvPKSgwcPokGDBtDX14enp6eiJ/qpZ8tLwsLCMGfOHFy9ehUSiQQSiQRhYWH45JNP0Lt3b6XzioqKYG1tjY0bNwIA+vTpAy8vL9SvXx8NGjTA/PnzYWRkhHPnzinOkUgkWLNmDXr27Al9fX04OTlh165dZeL/+eef0bFjR+jr6+Odd97B9evXceHCBbRu3RpGRkbo2bMnHj169Botrz7h4eHo2aMH3n//fdSzt8fYMWOgp6eHI0eOlHv8L7/8gtatWmHgwIGwt7fH8OHD4ezsjP379wMo7VXbu3cvhgwZAg8PDzg6OmLypElIS0vDmbNnAQAlJSVYu24dAgMC0KtXL9SpUwf17O3RqVOnKnvfQpHL5fgjagtadhsFhybdYFnLFZ6Dv0N+diru/PXbS8+VSrVgYGyleOgbmldR1ML78cgp9O/0Dvp1bA0nOxt8NdwbNXR18cupi+UeP3/kEHzY1QOu9rXhWMsawf4DIJfLEf13gtJxutraqGlqrHiYGL75F3lNIpfLsW9vOD4c4ot3PdrB0dEJEyZNQ3paGs6dPf3C8/aG78YHPXqi+/s9YG9fD6PHjIeenh6OHokAANRzcMTMWbPRpq0HatWqjRZu7vh4hD+iz59DSUmJ0rUMDY1gbmGheLwo0Rcbw4498Tj6BAounkJJ6kPkhIdCXlQI/XfK/xzTb90ZEgNDZG5ZhqK7NyDL+AdFt+NQnJT43zU790ZJVnppz/b9W5BlPMKTG3+iJD21qt6WRnsU8Tuuz16GlF9e/llJgEyuvkd1pJFJd0VFRkbi4cOH+P3337FkyRLMnj0bvXv3hrm5Oc6fP49Ro0Zh5MiRuH//frnn37t3DwMGDECfPn0QExODwMBATJ8+/YWvN3jwYEyaNAlNmjRBUlISkpKSMHjwYAQGBuLw4cNISkpSHHvgwAHk5+dj8ODBZa5TUlKCHTt2IC8vDx4eHkr7goKC4OPjg6tXr8LX1xdDhgzBtWvXlI6ZPXs2Zs2ahcuXL0NbWxtDhw7F1KlTsXz5cpw6dQo3b97UqPKJoqIi3Lh5U+mLlFQqhZubG67FxZV7zrW4OLi5uytta9WqleL45ORkZGRkwP2ZaxoaGsLV1RVx/7bXzZs3kZaWBolEgi/GjMFQX18EBQWV+WJVHeWk30d+ziPY1W+n2Kanbwzrus2Rcjfmpedm/XMXW+d1xPZvu+PY9snIyXio5mg1Q1FxMa7deYi2TVwU26RSKdo2dkbszcSXnPmfgsIiFJeUlEmqL8bdQrdx36D/jMVYsGUvMnOrV49iSnIyMjLS4eb23++soaEhGrg2RNy1v8s9p6ioCDdvXkcLt5aKbaWfCy0RH1f+OQCQl5cHAwMDaGlpKW1fu2Ylhg7xwcQvx+DokcOQV4f73lpa0LZzwJMbf/23TS7Hk5t/QcfepdxT9Bq3RNHdmzD2HoGas1bBcsJCGHj2ASQS5WPu34ap71hYBa2Gxbh50G/TRc1vhogEucd54MABGD136+/5XouKsLCwwIoVKyCVSuHq6opFixYhPz8fM2fOBADMmDED3377LaKiojBkyJAy569ZswbOzs5YvHgxAMDV1RV//PEHvvvuu3JfT19fH0ZGRtDW1lYqU2nXrh1cXV2xdetWTJ06FQAQGhqKQYMGKb3PP/74Ax4eHigoKICRkRHCw8PRuHFjpdcYNGgQAgMDAQDz5s3D0aNHsXLlSnz//feKYyZPnowPPvgAADB+/Hh89NFHOHbsGNq3bw8ACAgIQFhY2AvbrbCwEIWFhWW26em9uOTgTWRnZ0Mmk8HcXLnH1NzMDPfv3Sv3nIyMDJg/V3ZibmaGjIwMxX4A5V7z6b6k5NI62x9//BGffvopbGxssGfPHkybPh0bfvgBxsbGb/zeNFV+TumdDn0jS6Xt+sY1kZ/zzwvPs7ZvgS6DF8LMyhH52am49Ntq7FszDIMm7oNujepxu/5FMnPyUSKTlSkjsTA1xp3kit05WrHzEKzMTJQS93bNGqBrqyaoXdMC9x+lYdXuIxi7JAxhsz6HllTU/R4KGRnpAACz534fzczMFb+Pz8vOzir3c8HMzPyFnwtZWVn4v59+xAc9vZS2+w4bgeYt3KBXowauXL6INatX4PHjx+jbr//rviWNIDUwhkRLC7LcLKXtspxs6FrVLvccLQsr6Do3QkHMWWSGhkDL0gYm3iMg0dJG3m/himMM3u2K/FOHkXF8H3TqOMG478eQFxej4HKU2t8XVR9yLo6jEkE+8T09PRETE6P02LBhg8rXadKkCaTP/NGysbFBs2bNFM+1tLRgaWmJ1NTyb5ldu3YNbdu2Vdr2fM9zRQUGBipKWVJSUnDo0CF88onyQClXV1fExMTg/Pnz+PzzzzFixAj8/bdyj87zr+/h4VGmp7t58+aKn21sbABA6X3b2Ni88D0DwMKFC2Fqaqr0WLt2rQrvVhzkMhkAYPCQIejQoQPq16+PCRMnAigdpFmd3Li8HxtntVQ8ZCXFr3Ud+4ad4Ny8ByxruaKua0f0/GQ9nhRk41bs4UqOuPoJ/fUEIqJjETJ2GPR0dBTbP2jbAp3dG6N+XVt4tmyC5eNH4K/b93Ex7sUD4TTdiePHMGhAH8Wj+DX/vakiPz8Pc2fPQl37ehjqO1xp35Chw9C4SVM4O7tg4KAhGDDwQ4Tv3qn2mDSSRAJZXjayd29E8YM7KIw9j7zj+6Dftuszx0hR9PAuciN2ovjhXTyOPo7H0Seg/27XF1+XiN6YID3dhoaGcHFRvjX2bAmIVCotc2uwqKiozHV0nvnDBpTWRJe3TfZv8qVOw4cPx/Tp03H27FmcOXMGjo6O6Nixo9Ixurq6ivfdqlUrXLhwAcuXL8e6detUeq1n36Pk31uGz2972XueMWMGJv6bfD714AUlOJXBxMQEUqm0TI9XRmYmzC0syj3H3NwcGf9O26h0/L+9Yk//m5GRAYtnrpGRmQlnJycAUGy3t7dX7NfV0UEtW1ukaljN+5uq19gTA+3/+zJWUvwEAPA4Nw2GJtaK7Y9z/oFl7UYVvq6evglMazogK+1u5QWrocyMDaAllSI9O1dpe3pWDixNXn5XZMuh3xH660msnRKABnVrvfTYOtYWMDMyxL2UNLRtXH6JgKZr09ZDMcMI8N/nc2ZGBiws/ru7kpmZAScn53KvYWJiWu7nQmZmBswtlHu/8/PzMTtoJvQN9PFV0NfQfsVAVFfXRvi/n35EUdET6OiIt7Zblp8DeUkJpEamStulxiYoycks/5ycLMhLipWmlShOfQgtEzNASwsoKYEsJxMlKQ+UzitOfQi9pq0r+y0Q0TM08t6mlZUVkpOTlRJvVafoq4hGjRohOjpaaduzAxvLo6urW24pjKWlJby9vREaGoqwsDD4+/u/8vVlMlmZMo/nX//cuXNo1KjiSVJF6OnpwcTEROmhrtISoPQLQX0XF8Rc/W+6KplMhpiYGDRq2LDccxo1bFjm//mVK1cUx9va2sLc3Fzpmnn5+YiPj0fDf9vLpX596OjoKH2hKC4uRkpqKqytrVGd6NYwgmnNeoqHuY0LDIyt8ODGWcUxTwpykXovFjb13Cp83aLCPGSn3YOBsZUaotYsOtraaORQW2kQpEwmQ/S1BDR3sX/heWEHT2LD/kismuSPxo4vnx4PAFLSs5CVlw8rM/GWNxkYGKB2bTvFw96+HszNLXD16hXFMfn5ebgeH4eGjRqXew0dHR24uDRA7DPnyGQyXI25AteG/52Tn5+H4FnToa2tjVnBcys0QPLWrZswMjIWdcINACgpQfGDO9B1eaYNJRLoujRBUeLNck8punMd2pY2SjXcWjVtUZKdAfz7t+vJnevQslL+cqhV0xYlmWmV/x6ISEEj563q0qULHj16hEWLFmHgwIE4fPgwDh06BBMTk0p9nVGjRmHx4sWYMmUKAgMDcenSpZfWQgOAg4MDbt++jZiYGNSpUwfGxsaKhDUwMBC9e/dGSUkJRowYoXTejBkz0LNnT9jb2yMnJwfbt2/HiRMnEBERoXTczp070bp1a3To0AE//vgjoqOjFTOgiFn//v2xeMkS1K9fH64NGmDvL7+gsLAQ771XOr9uSEgILC0tFV9W+vXrh6nTpmH3nj1o8847OHnyJG7cuIFxY8cCKO3N9/b2xo4dO2BXuzZsbGywdetWWFpaot2/JTqGBgbw8vLC1m3bUNPKCjbW1orZYDp26CBAK1QdiUSCZh2G43LkWpjWdICxhR0uHlkBAxNrODTprjhu/3o/ODbpjqbthwEAzh74DvUaecLYvDbyslNx8egqSKRSuLj1ftFLVSu+73fE7A070djBDk2c6mL7kdN4XPgEfTu0AgAE/fAzrM1MMHZQDwBA2K8nsWbvUSwYOQS1a5rjn6wcAICBni4Maughv6AQ6345hm6tm6KmqTHupaZh+c+HUNfaAh5NGwj2PiubRCJBX+/++L8d21G7th1sbGph29YwWFha4l2P9orjvpoxBR7t2qN3H28AgHd/Hyxdsggu9RugQQNX/PJLOAoKC9D9vdIxK/n5eQj+ajoKCwsxacp0PM7Px+P80qnvTExNoaWlhejzZ5GRkYGGDRtBR1cXMVcuY+f/7UB/n4Fl4hSjvFOHYPrhZyi6fxtF92/BoMMHkOjooeDi7wAAkw9HQpadgdzDPwMA8s8dg36792DcZxjyzxyFVk0bGHr2xePT/80UlR91GBajg2Hg2QeFseehU9cZBm09kb17U7kxvG20DA1g+MwXbQPHOjBp0RBP0rNQcC/pJWe+farDeOWqpJFJd6NGjfD9999jwYIFmDdvHnx8fDB58mSsX7++Ul/H3t4eu3fvxoQJE7By5Uq0adMGCxYsKFOL/SwfHx/s2bMHnp6eyMzMRGhoKPz8/AAA3bt3R61atdCkSRPUrq08yCU1NRXDhw9HUlISTE1N0bx5c0RERCiSzqfmzJmDHTt2YPTo0ahVqxZ++umnMoMtxahz587Iys7Gtq1bkZ6RAWcnJ8ybO1dRJpL66BEkz9TnN27cGNOmTsXmLVsQFhYGOzs7BAUFKU0bOWjgQBQUFGDFypXIzc1FkyZNMG+uck9YYEAAtLS0EBISgsLCQjR0dcW3CxdW60GUT7XoEoiiJ4/x++5gPCnIhq1DK3gF/KA0R3d2WiIK8v67vZ+XlYJj2yehID8T+kYWsHVoBe8x/wd9o/LLgKqbD9o2R0ZOLtbs/Q1pWTlwta+FVRP9YWla+u8lOS0T0md6EHceP4ei4hJMWf2j0nU+69cNo7y7QyqV4sa9ZBw4fRk5+QWwMjPGu03rY3T/96Cro5Efv6/NZ+BgFBQUYNXKZcjLzUXjJk0xZ+5Cpd/H5KQkZGf9N7d0x85dkJWdiR+3bi5dNMvJGXPmLlB8LiTcvIn4+NIZiz4LUO7I2BC6FTY2ttDS0sbBA/uw8Ye1kMvlqFW7NgI+HYkPeigPthSrwtjzyDE0htH7PqWL4zxMRMam/0GWW9qOWmaWSpmPLCsdmRsXwaiPLyy/nI+S7Azkn45A/okDimOK799G5pblMOrxIYy6eaMk4xFy9m9DQcyZKn9/msi0VVN4HNuqeN44pHRyhntb9iA2YIZQYWkksU7tl56ejrFjx2L//v2QSqXw8fHB8uXLy0zyUR65XA4vLy8cPnwY4eHh8Pb2rvDrSuTVYl4lzZCbmws7OzuEhoZiwIABKp8vkUhU/h9YWW4lJLz6IKqQvX86CR1CtTHSKlzoEKqFBzas1a0spuu/EjqEauPikuhXH0Sv1KsoXrDXDjuhvmv7dVHftXv27ImkpCSsW7cORUVF8Pf3xzvvvIPt27e/8tylS5fi6NGjOHTokMo5W/XqahGITCbDP//8g8WLF8PMzAx9+/YVOiQiIiIitRJjt+21a9dw+PBhxaKCALBy5Up4eXkhJCSkTKXCs2JiYrB48WJcvHgRtWq9fNB8eZh0V4LExEQ4OjqiTp06CAsLe+XIeiIiIiJ6sfLWFNHT03vjiR/Onj0LMzMzRcINlJYHS6VSnD9/Hv37lz+/f35+PoYOHYrVq1crrdWiCo2cvURsHBwcIJfLce/ePXTr1u21ryOXywUpLSEiIiLSJOWtKbJw4cI3vm5ycnKZGcy0tbVhYWGB5H8X1SvPhAkT0K5dO/Tr1++1X5tdskRERESkUcpbU+RlvdzTp09/4YriTz2/2GBF7du3D5GRkbhy5cqrD34JJt1EREREpDJ11nSrWkoyadIkxWxyL+Lk5ARbW9syq3YXFxcjPT39hWUjkZGRSEhIgJmZmdJ2Hx8fdOzYESdOnKhQjEy6iYiIiEjUrKysYGX16oXcPDw8kJmZiUuXLqFVq9I1GCIjIyGTydC2bdtyz5k+fToCAwOVtjVr1gxLly5Fnz59Khwjk24iIiIiUpkY5+lu1KgRevTogU8//RRr165FUVERxowZgyFDhihmLnnw4AG6deuGLVu2oE2bNrC1tS23F9ze3h6Ojo4Vfm0OpCQiIiIilcnl6nuo048//oiGDRuiW7du8PLyQocOHZQWYCwqKkJ8fDzy/10Bt7Kwp5uIiIiI3hoWFhYvXQjn6ax0L/M6a0uyp5uIiIiISM2YdBMRERERqRnLS4iIiIhIZTKZ0BGIC3u6iYiIiIjUjD3dRERERKQydc8yUt2wp5uIiIiISM3Y001EREREKmNPt2rY001EREREpGZMuomIiIiI1IzlJURERESkMhnLS1TCnm4iIiIiIjVjTzcRERERqUyu1pGUEjVeWxjs6SYiIiIiUjP2dBMRERGRyjhloGrY001EREREpGZMuomIiIiI1IzlJURERESkMplM6AjEhT3dRERERERqxp5uIiIiIlIZB1Kqhj3dRERERERqxp5uIiIiIlIZl4FXDXu6iYiIiIjUjEk3EREREZGaMekmIiIiIlIziVzOsackDoWFhVi4cCFmzJgBPT09ocMRLbZj5WFbVh62ZeVgO1YetiVVNibdJBrZ2dkwNTVFVlYWTExMhA5HtNiOlYdtWXnYlpWD7Vh52JZU2VheQkRERESkZky6iYiIiIjUjEk3EREREZGaMekm0dDT08Ps2bM5oOUNsR0rD9uy8rAtKwfbsfKwLamycSAlEREREZGasaebiIiIiEjNmHQTEREREakZk24iIiIiIjVj0k1EREREpGZMuomIiIiI1IxJNxERERGRmmkLHQARVb2nM4VKJBKBI6G3yb59+yp8bN++fdUYCRFR1eM83aTx4uPjsXLlSly7dg0A0KhRI4wdOxaurq4CRyY+GzduxNKlS3Hjxg0AQP369fHll18iMDBQ4Mg024ABAyp87J49e9QYibhJpco3VyUSCZ79E/Tsl8CSkpIqi6s6KCgowMqVK3H8+HGkpqZCJpMp7b98+bJAkYlLWloagoODX9iO6enpAkVG1QF7ukmj7d69G0OGDEHr1q3h4eEBADh37hyaNm2KHTt2wMfHR+AIxSM4OBhLlizB2LFjFW159uxZTJgwAYmJiZg7d67AEWouU1NToUOoFp5NYH777TdMmzYNCxYsUPr3OGvWLCxYsECoEEUrICAAR44cwcCBA9GmTRvexXpNH3/8MW7evImAgADY2NiwHalSsaebNJqzszN8fX3LJISzZ8/Gtm3bkJCQIFBk4mNlZYUVK1bgo48+Utr+008/YezYsfjnn38EiozeRk2bNsXatWvRoUMHpe2nTp3CZ599prizRRVjamqKgwcPon379kKHImrGxsaIiopCixYthA6FqiEOpCSNlpSUhOHDh5fZPmzYMCQlJQkQkXgVFRWhdevWZba3atUKxcXFAkREb7OEhASYmZmV2W5qaoo7d+5UeTxiZ2dnB2NjY6HDEL2GDRvi8ePHQodB1RR7ukmjeXl5YdCgQfD391faHhoaih07diAiIkKgyMRn7Nix0NHRwZIlS5S2T548GY8fP8bq1asFikzzubu7V/g2M2tnK6ZTp06oUaMGtm7dChsbGwBASkoKhg8fjoKCApw8eVLgCMXl0KFDWLFiBdauXYt69eoJHY5oXbhwAdOnT0dwcDCaNm0KHR0dpf0mJiYCRUbVAWu6SaP17dsX06ZNw6VLl/Duu+8CKK3p3rlzJ+bMmaM0GwJnO3i1jRs34siRI4q2PH/+PBITEzF8+HBMnDhRcdzzifnbztvbW+gQqp1Nmzahf//+sLe3R926dQEA9+7dQ/369bF3715hgxOh1q1bo6CgAE5OTjAwMCiTLHIAYMWYmZkhOzsbXbt2Vdoul8shkUg4wJfeCHu6SaM9P9vBi/DD8NU8PT0rdJxEIkFkZKSaoyEqTWSOHj2KuLg4AKUzE3Xv3p2D115D9+7dkZiY+MIBgCNGjBAoMnFp06YNtLW1MX78+HLbsXPnzgJFRtUBk24iIhJUQUEB9PT0mGy/AQMDA5w9e5YDAN+QgYEBrly5wilpSS04kJKISAUlJSUICQlBmzZtYGtrCwsLC6UHVYxMJsO8efNgZ2cHIyMj3L59GwAQFBSEjRs3Chyd+HAAYOVo3bo17t27J3QYVE2xpps03oULF164UAFrjyuOi2dUjjlz5mDDhg2YNGkSZs2aha+++gp37tzB3r17ERwcLHR4ovHNN99g8+bNWLRoET799FPF9qZNm2LZsmUICAgQMDrx+fbbbzFp0iTMnz8fzZo14wDA1zR27FiMHz8eU6ZMKbcdmzdvLlBkVB2wvIQ02oIFCzBr1iy4urqWqa9j7bFqfH19FYtnlFerOHv2bIEiExdnZ2esWLECvXr1grGxMWJiYhTbzp07h+3btwsdoii4uLhg3bp16NatG4yNjXH16lU4OTkhLi4OHh4eyMjIEDpEUXk6/uX532sOAFRNeeOInq6cynakN8WebtJoy5cvx6ZNm+Dn5yd0KKJ34MABLp5RCZKTk9GsWTMAgJGREbKysgAAvXv3RlBQkJChicqDBw/g4uJSZrtMJkNRUZEAEYnb8ePHhQ6hWnha5kSkDky6SaNJpVImiZWEi2dUjjp16iApKQn29vZwdnbGkSNH0LJlS1y4cAF6enpChycajRs3xqlTp8rMKb1r1y64u7sLFJV4cVaNysE5zkmdmHSTRpswYQJWr16NZcuWCR2K6C1evBjTpk3j4hlvqH///jh27Bjatm2LsWPHYtiwYdi4cSMSExMxYcIEocMTjeDgYIwYMQIPHjyATCbDnj17EB8fjy1btuDAgQNChyda+fn5SExMxJMnT5S2sxa54uLj47Fy5Upcu3YNQOlUlmPHjuWMJvTGWNNNGk0mk6FXr164fv06GjduXGZQy549ewSKTHwePXqEDz/8EL///jsXz6hE586dw5kzZ1C/fn306dNH6HBE5dSpU5g7dy6uXr2K3NxctGzZEsHBwXj//feFDk10Hj16BH9/fxw6dKjc/axFrpjdu3djyJAhaN26NTw8PACU/o5fuHABO3bsgI+Pj8ARkpgx6SaNNmbMGGzYsAGenp7lDv4LDQ0VKDLx4eIZr69ly5Y4duwYzM3NMXfuXEyePBkGBgZCh0Wk4Ovri7t372LZsmXo0qULwsPDkZKSgm+++QaLFy9Gr169hA5RFJydneHr64u5c+cqbZ89eza2bduGhIQEgSKj6oBJN2k0Y2Nj7Nixg38wKgEXz3h9+vr6uHHjBurUqQMtLS0kJSXB2tpa6LBELzMzE7t27cKtW7cwefJkWFhY4PLly7CxsYGdnZ3Q4YlKrVq18Msvv6BNmzYwMTHBxYsX0aBBA+zbtw+LFi1CVFSU0CGKgoGBAWJjY8sM8r1x4wZatGiB/Px8gSKj6oA13aTRLCws4OzsLHQY1QIXz3h9bm5u8Pf3R4cOHSCXyxESEgIjI6Nyj+Vc3RUTGxuL7t27w9TUFHfu3EFgYCAsLCywZ88eJCYmYsuWLUKHKCp5eXmKL4Lm5uZ49OgRGjRogGbNmnEOfhV06dIFp06dKpN0R0VFoWPHjgJFRdUFk27SaF9//TVmz56N0NBQ3s5/Q1w84/WFhYVh9uzZOHDgACQSCQ4dOgRt7bIfnxKJhEl3BU2cOBF+fn5YtGiR0qw6Xl5eGDp0qICRiZOrqyvi4+Ph4OCAFi1aYN26dXBwcMDatWtRq1YtocMTjb59+2LatGm4dOkS3n33XQClNd07d+7EnDlzsG/fPqVjiVTB8hLSaO7u7khISIBcLoeDg0OZRJE9OBXHxTMqh1QqRXJyMstL3pCpqSkuX74MZ2dnpcVx7t69C1dXVxQUFAgdoqhs27YNxcXF8PPzw6VLl9CjRw+kp6dDV1cXYWFhGDx4sNAhikJ5i+OUh5+Z9DrY000azdvbW+gQqg0unlE5ZDKZ0CFUC3p6esjOzi6z/fr167CyshIgInEbNmyY4udWrVrh7t27iIuLg729PWrWrClgZOLC329SJ/Z0ExGp6MaNGzh+/DhSU1PL/JFmeUnFBAYGIi0tDT///DMsLCwQGxsLLS0teHt7o1OnTpybn4iqHSbdpPGeznCQkJCAKVOmcIaDN3Dq1CmsW7cOt27dws6dO2FnZ4etW7fC0dERHTp0EDo8Ufjhhx/w+eefo2bNmrC1tVUq15FIJCx5qqCsrCwMHDgQFy9eRE5ODmrXro3k5GR4eHjg4MGDMDQ0FDpE0UhKSsKaNWsQFRWFpKQkSKVSODk5wdvbG35+ftDS0hI6RFG4du0azp07Bw8PDzRs2BBxcXFYvnw5CgsLMWzYMHTt2lXoEEnkmHSTRnt+hoP4+Hg4OTlh1qxZnOFARbt378bHH38MX19fbN26FX///TecnJywatUqHDx4EAcPHhQ6RFGoV68eRo8ejWnTpgkdSrUQFRWF2NhYxeI43bt3FzokUbl48SK6d+8OFxcX6Ovr4+zZsxg6dCiePHmCiIgING7cGIcPH1YarEplHT58GP369YORkRHy8/MRHh6O4cOHo0WLFpDJZDh58iSOHDnCxJvejJxIg3Xr1k0+ZcoUuVwulxsZGckTEhLkcrlcfvr0aXm9evUEjEx83Nzc5Js3b5bL5cptefnyZbmNjY2QoYmKsbGxou2IhNa+fXv5119/rXi+detWedu2beVyuVyenp4ud3Nzk48bN06o8ETDw8ND/tVXX8nlcrn8p59+kpubm8tnzpyp2D99+nT5e++9J1R4VE1UbJgukUAuXLiAkSNHltluZ2eH5ORkASISr/j4eHTq1KnMdlNTU2RmZlZ9QCI1aNAgHDlyROgwqoVjx46hd+/ecHZ2hrOzM3r37o3ffvtN6LBE5fLly/j4448Vz4cOHYrLly8jJSUF5ubmWLRoEXbt2iVghOLw119/wc/PDwDw4YcfIicnBwMHDlTs9/X1RWxsrEDRUXXB2UtIo3GGg8pja2uLmzdvwsHBQWl7VFQUnJychAlKhFxcXBAUFIRz586VO9/5uHHjBIpMXL7//nuMHz8eAwcOxPjx4wGUzofs5eWFpUuX4osvvhA4QnGwtrZGUlKS4nc4JSUFxcXFinn369evj/T0dCFDFI2n4zOkUilq1KgBU1NTxT5jY2NkZWUJFRpVE0y6SSMlJiaiTp066Nu3L+bOnYuff/4ZQOmHYmJiIqZNmwYfHx+BoxSHLVu2YPDgwfj0008xfvx4bNq0CRKJBA8fPsTZs2cxefJkBAUFCR2maKxfvx5GRkY4efIkTp48qbRPIpEw6a6gBQsWYOnSpRgzZoxi27hx49C+fXssWLCASXcFeXt7Y9SoUfjf//4HPT09zJs3D507d4a+vj6A0jtcHHD+ag4ODrhx44ZiBeSzZ8/C3t5esT8xMZGLDNEb40BK0khaWlpISkqCnp4eZzh4Q0/b0srKCgsWLMDChQuRn58PoPROwuTJkzFv3jyBo6S3jZGREWJiYsost33jxg24u7sjNzdXoMjEJTc3FwEBAdizZw9KSkrg4eGBbdu2wdHREQBw5MgRZGVlYdCgQQJHqtnWrl2LunXrolevXuXunzlzJlJTU7Fhw4YqjoyqEybdpJGeX/WPMxy8vufb8smTJ7h58yZyc3PRuHFjGBkZCRwhvY2GDh0Kd3d3TJkyRWl7SEgILl68iB07dggUmTgVFBSguLiYv89EGoxJN2kkqVSKlJQU1m1XArblm5s4cSLmzZsHQ0NDTJw48aXHLlmypIqiErdvvvkGISEhaN++PTw8PACU1nSfPn0akyZNUtQkA6yTr4jQ0FAMGTJEUVZCryc0NBSDBw+GgYGB0KFQNcSkmzSSVCrFZ5999soPPiY4ryaVStG0aVNoa798CAcXdXkxT09PhIeHw8zMDJ6eni88TiKRIDIysgojE6+n5Q+vIpFIcOvWLTVHI342NjZ4/PgxBg0ahICAALRr107okESJ7UjqxIGUpLH++OMP6OrqvnD/sysB0st98MEHvO38Bo4fP17uz/T6bt++LXQI1cqDBw+wf/9+hIWFoUuXLnBycoK/vz9GjBgBW1tbocMTDbYjqRN7ukkjPV+HTK+PbUliUFxcjIKCAn45rAQpKSnYtm0bNm/ejLi4OPTo0QMBAQHo06cPpFIuz1FRbEeqbPxXQxqJvdiVh21Z+S5evIipU6diyJAhGDBggNKDXu5pL+Kz5s+fDyMjI5iZmeH9999HRkaGMMFVEzY2NujQoQM8PDwglUrxxx9/YMSIEXB2dsaJEyeEDk802I5U2Zh0k0biDZjKw7asXDt27EC7du1w7do1hIeHo6ioCH/99RciIyOVFtOg8i1ZsgR5eXmK52fOnEFwcDCCgoLw888/4969e5zC8jWlpKQgJCQETZo0QZcuXZCdnY0DBw7g9u3bePDgAT788EOMGDFC6DA1HtuR1IXlJaSRNm/ejCFDhkBPT0/oUETv7t27sLe3R2FhIWrUqFHuMUlJSVz4oYKaN2+OkSNH4osvvoCxsTGuXr0KR0dHjBw5ErVq1cKcOXOEDlGjWVtbIyIiAu7u7gBKZ4b5+++/cfjwYQDAwYMHMX78eNy4cUPIMEWnT58+iIiIQIMGDRAYGIjhw4fDwsJC6ZjU1FTY2tpCJpMJFKXmYzuSOnEgJWmkZ3sRbty4gePHjyM1NbXMh1xwcHBVhyY69erVAwC0bNkS27dvh5ubm9L+3bt3Y9SoUXj06JEA0YlPQkKCYgENXV1d5OXlQSKRYMKECejatSuT7lfIycmBpaWl4nlUVJTSwi1NmjTBw4cPhQhN1KytrXHy5EnF9IvlsbKy4gDWV2A7kjqxvIQ02g8//IBGjRohODgYu3btQnh4uOKxd+9eocMTlS5duuDdd9/Fd999BwDIy8uDn58fPv74Y8ycOVPg6MTD3NwcOTk5AAA7Ozv8+eefAIDMzEzFSp/0YnZ2drh27RqA0tUUr169qjQtW1paGudIfg2dO3dGy5Yty2x/8uQJtmzZAqB0fMfTL+FUPrYjqRPLS0ij1atXD6NHj8a0adOEDqVa+PXXXxEYGAgXFxckJSXByMgI27ZtQ9OmTYUOTTSGDh2K1q1bKxbMWblyJfr164ejR4/C3d0d4eHhQoeo0WbMmIG9e/di5syZOHjwIM6cOYNbt25BS0sLALB+/Xps2bIFUVFRAkcqLlpaWkhKSiozS1FaWhqsra1RUlIiUGTiwnYkdWJ5CWm0jIwMpVvP9GZ69uyJAQMGYM2aNdDW1sb+/fuZcKto1apVKCgoAAB89dVX0NHRwZkzZ+Dj44PJkycLHJ3mCw4OxoMHDzBu3DjY2tpi27ZtioQbAH766Sf06dNHwAjFSS6XlztT0f379znAVwVsR1InJt2k0QYNGoQjR45g1KhRQociegkJCRg6dCiSk5MRERGBkydPom/fvhg/fjzmz58PHR0doUMUhWcHVUmlUkyfPh0FBQVYvXo13N3dkZycLGB0mk9fX19xm748XHxINe7u7pBIJJBIJOjWrZvSyrMlJSW4ffs2evToIWCE4sB2pKrApJs0mouLC4KCgnDu3Dk0a9asTGI4btw4gSITHzc3N/Tq1QsREREwMzPDe++9By8vLwwfPhxHjx7FlStXhA5RoxUWFuLrr7/G0aNHoauri6lTp8Lb2xuhoaGYNWsWtLS0MGHCBKHDFI2uXbtiz549MDMzU9qenZ0Nb29vREZGChOYyHh7ewMAYmJiyqw8q6urCwcHB/j4+AgUnXiwHakqsKabNJqjo+ML90kkEty6dasKoxG3rVu34uOPPy6zPScnB19++SU2btwoQFTiMW3aNKxbtw7du3fHmTNn8OjRI/j7++PcuXOYOXMmBg0apFQmQS/3opVSU1NTYWdnh6KiIoEiE6fNmzdj8ODBL5wWlCqG7UjqxKSbiKgCnJycsGzZMvTt2xd//vknmjdvDj8/P2zcuJGrfqogNjYWQOmdl8jISKVynZKSEhw+fBjr1q3DnTt3BIqQiEg9mHQTvWX+/vtvJCYm4smTJ4ptEomEg9deQVdXF7dv34adnR2A0trk6OhoNGvWTODIxEUqlSq+pJT350dfXx8rV67EJ598UtWhiY6FhQWuX7+OmjVrwtzc/KVf/tLT06swMnFhO1JVYU03aZynU7EZGhpi4sSJLz12yZIlVRSV+N26dQv9+/fHH3/8AYlEokh4nv6B4VRYL1dSUgJdXV3Fc21tbaW6T6qY27dvQy6Xw8nJCdHR0bCyslLs09XVhbW1Nct0Kmjp0qUwNjZW/Mw7Lq+H7UhVhT3dpHE8PT0RHh4OMzMzeHp6vvRYznRQcX369IGWlhY2bNgAR0dHREdHIy0tDZMmTUJISAg6duwodIgaTSqVomfPntDT0wMA7N+/H127doWhoaHScXv27BEiPCIi0nBMuoneEjVr1kRkZCSaN28OU1NTREdHw9XVFZGRkZg0aRJnL3kFf3//Ch0XGhqq5kiqh82bN6NmzZro1asXAGDq1KlYv349GjdujJ9++okr/lVAdnZ2hY81MTFRYyTixnakqsKkmzRSReo5JRIJZ9xQgbm5OS5fvgxHR0c4Oztjw4YN8PT0REJCApo1a8YlzKlKubq6Ys2aNejatSvOnj2Lbt26YdmyZThw4AC0tbV5x6ACnq2PfxWWj70Y25GqCmu6SSOFhYWhXr16cHd3L3ewFamuadOmuHr1KhwdHdG2bVssWrQIurq6WL9+PZycnIQOj94y9+7dg4uLCwBg7969GDhwID777DO0b98eXbp0ETY4kXi2vO7OnTuYPn06/Pz84OHhAQA4e/YsNm/ejIULFwoVoiiwHamqsKebNNIXX3yhuMXs7++PYcOGKU0tRqqLiIhAXl4eBgwYgBs3bqBPnz64fv06LC0tsWPHDnTr1k3oEOktYm1tjYiICLi7u8Pd3R0TJ07Exx9/jISEBLRo0QK5ublChygq3bp1Q2BgID766COl7du3b8f69etx4sQJYQITGbYjqROTbtJYhYWF2LNnDzZt2oQzZ86gV69eCAgIwPvvv8/R5ZUkPT39lVNkEamDr68v4uLi4O7ujp9++gmJiYmwtLTEvn37MHPmTPz5559ChygqBgYGuHr1KurXr6+0/fr163Bzc2P5WAWxHUmdWF5CGktPTw8fffQRPvroI9y9exdhYWEYPXo0iouL8ddff3G6tgqq6HzHmzZtUnMkRP9ZvXo1Zs2ahXv37mH37t2wtLQEAFy6dKlMLyO9Wt26dfHDDz9g0aJFSts3bNiAunXrChSV+LAdSZ2YdJMoPB3oIpfLOZBFRayPJ01kZmaGVatWldk+Z84cAaIRv6VLl8LHxweHDh1C27ZtAQDR0dG4ceMGdu/eLXB04sF2JHVieQlprGfLS6KiotC7d2/4+/ujR48ekEqlQocnGqyPJ030+++/v3R/p06dqiiS6uPevXtYs2YN4uLiAACNGjXCqFGj2EOrIrYjqQuTbtJIo0ePxo4dO1C3bl188skn8PX1Rc2aNYUOS7RYH0+aprwvzs/+W+QdLSKqbph0k0aSSqWwt7eHu7v7S5NCzuWruqf18Vu2bGF9PAkmKytL6XlRURGuXLmCoKAgzJ8/n7PpVEBsbCyaNm0KqVSK2NjYlx7bvHnzKopKfNiOVFVY000aafjw4eyBVRPWx5MmMDU1LbPtvffeg66uLiZOnIhLly4JEJW4uLm5ITk5GdbW1nBzc1P8Xj9PIpHwd/0l2I5UVZh0k0YKCwsTOoRqpbz6+FWrVrE+njSOjY0N4uPjhQ5DFG7fvg0rKyvFz/R62I5UVVheQlTNsT6eNNHzt/HlcjmSkpLw7bffori4GFFRUQJFRkSkHky6iao51seTJnq2zOlZ7777LjZt2oSGDRsKFJl4PXz4EFFRUUhNTYVMJlPaN27cOIGiEh+2I6kLk26ias7Pz69C9fGhoaFVEA1Rqbt37yo9l0qlsLKyQo0aNQSKSNzCwsIwcuRI6OrqwtLSUul3XiKR4NatWwJGJx5sR1InJt1EREQiV7duXYwaNQozZszgOI03wHYkdeJASiIiqjKPHz/GsWPH0Lt3bwDAjBkzUFhYqNivpaWFefPmscdbRfn5+RgyZAgTxTfEdiR14r8qIiKqMps3b8a6desUz1etWoUzZ87gypUruHLlCrZt24Y1a9YIGKE4BQQEYOfOnUKHIXpsR1InlpcQEVGV6dixI6ZOnYo+ffoAAIyNjXH16lU4OTkBALZt24bVq1fj7NmzQoYpOiUlJejduzceP36MZs2aQUdHR2n/kiVLBIpMXNiOpE4sLyEioipz8+ZNNGvWTPG8Ro0aSrfy27Rpgy+++EKI0ERt4cKFiIiIgKurKwCUGQBIFcN2JHViTzcREVUZfX19xMTEKJKa58XFxcHNzQ0FBQVVHJm4mZubY+nSpfDz8xM6FFFjO5I6saabiIiqTJ06dfDnn3++cH9sbCzq1KlThRFVD3p6emjfvr3QYYge25HUiUk3ERFVGS8vLwQHB5fbk/348WPMmTMHvXr1EiAycRs/fjxWrlwpdBiix3YkdWJ5CRERVZmUlBS4ublBV1cXY8aMQYMGDQAA8fHxWLVqFYqLi3HlyhXY2NgIHKm49O/fH5GRkbC0tESTJk3KDADkirMVw3YkdeJASiIiqjI2NjY4c+YMPv/8c0yfPl2xDLxEIsF7772H77//ngn3azAzM8OAAQOEDkP02I6kTuzpJiIiQaSnp+PmzZsAABcXF1hYWAgcERGR+jDpJiIiIiJSM5aXEBFRlVDltj1rZ1Xj6Oj40nmkb926VYXRiBfbkdSJSTcREVUJU1NTxc9yuRzh4eEwNTVF69atAQCXLl1CZmYma2pfw5dffqn0vKioCFeuXMHhw4cxZcoUYYISIbYjqRPLS4iIqMpNmzYN6enpWLt2LbS0tACULsE9evRomJiY4H//+5/AEVYPq1evxsWLFxEaGip0KKLGdqTKwKSbiIiqnJWVFaKiosqsTBkfH4927dohLS1NoMiql1u3bsHNzQ3Z2dlChyJqbEeqDFwch4iIqlxxcTHi4uLKbI+Li4NMJhMgoupp165dnBWmErAdqTKwppuIiKqcv78/AgICkJCQgDZt2gAAzp8/j2+//Rb+/v4CRyc+7u7uSgMA5XI5kpOT8ejRI3z//fcCRiYubEdSJybdRERU5UJCQmBra4vFixcjKSkJAFCrVi1MmTIFkyZNEjg68enXr59SsiiVSmFlZYUuXbqgYcOGAkYmLmxHUifWdBMRkaCe1smamJgIHIn4VLTGmG37cmxHqgpMuomIiERKKpW+dF5puVwOiUSCkpKSKoxKfNiOVBVYXkJERFUuJSUFkydPxrFjx5Camorn+3+Y3FTM8ePHFT/L5XJ4eXlhw4YNsLOzEzAq8WE7UlVgTzcREVW5nj17IjExEWPGjEGtWrXK9DL269dPoMjEzdjYGFevXoWTk5PQoYga25HUgT3dRERU5aKionDq1Cm4ubkJHQoRUZXgPN1ERFTl6tatW6akhIioOmPSTUREVW7ZsmWYPn067ty5I3Qo1c7LBgRSxbEdqbKxppuIiKqcubk58vPzUVxcDAMDA+jo6CjtT09PFygycRkwYIDS8/3796Nr164wNDRU2r5nz56qDEt02I5UFVjTTUREVW7ZsmVCh1AtmJqaKj0fNmyYQJGIG9uRqgJ7uomIiIiI1Iw93UREVCWys7MVK/q9agVArvxHRNUNe7qJiKhKaGlpISkpCdbW1i9cAZAr/xFRdcWebiIiqhKRkZHIysqCtbW10gqARERvA/Z0ExFRlZFKpahXrx48PT0Vjzp16ggdFhGR2jHpJiKiKnPixAnF4/z583jy5AmcnJzQtWtXRRJuY2MjdJhERJWOSTcREQmioKAAZ86cUSTh0dHRKCoqQsOGDfHXX38JHR4RUaVi0k1ERIJ68uQJTp8+jUOHDmHdunXIzc3lQEoiqnaYdBMRUZV68uQJzp07h+PHjyvKTOrWrYtOnTqhU6dO6Ny5M+zt7YUOk4ioUjHpJiKiKtO1a1ecP38ejo6O6Ny5Mzp27IjOnTujVq1aQodGRKRWTLqJiKjK6OjooFatWvD29kaXLl3QuXNnWFpaCh0WEZHaMekmIqIqk5eXh1OnTuHEiRM4fvw4YmJi0KBBA3Tu3FmRhFtZWQkdJhFRpWPSTUREgsnJyUFUVJSivvvq1auoX78+/vzzT6FDIyKqVFKhAyAioreXoaEhLCwsYGFhAXNzc2hra+PatWtCh0VEVOnY001ERFVGJpPh4sWLivKS06dPIy8vD3Z2dkqrVNarV0/oUImIKhWTbiIiqjImJibIy8uDra2tIsHu0qULnJ2dhQ6NiEitmHQTEVGVWbduHTw9PdGgQQOhQyEiqlJMuomIiIiI1IwDKYmIiIiI1IxJNxERERGRmjHpJiIiIiJSMybdRERERERqxqSbiIiIiEjNmHQTEREREakZk24iIiIiIjX7f8/IUI6ndCJ8AAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Now a pairwise correlation using a heatmap of selected numerical features\n",
    "\n",
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(df[selected_numerical_features].corr(), annot = True, cmap = 'coolwarm', center=0)\n",
    "plt.title('Pairwise correlation')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "0d633fe5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 1000x600 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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7y/Kly8vDAwMhKurq8bjMIT432My5s2bl2OdedUghBBpaWli2rRpwsHBQejr6wtbW1sxceJEkZycnKPt8uXLBQAxZMgQjene3t4CgAgNDc2xTG7s7OwEAAFAKBQKYWpqKmrWrCm++OILceLEiVyXebX/U1JSxLhx44S7u7soU6aMMDExEe7u7tLjWLIlJSWJXr16CbVaLQBIx0b24x9ye6xDXo87qVmzpjh9+rTw8PAQhoaGws7OTixbtizH8tHR0cLb21solUphaWkpJk2aJPbv359jnXnVltvjToR4t2PtTcf2qwqyn9natm0rAIi//vrrjevOzeuPO8m2adMmUadOHaFUKoW5ubno3bu3uHv3rkab7J+f6Oho8fHHHwtjY2NhaWkppk6dqvF4kjfta/Yx+PqrVatWUrtly5aJ6tWrC319fWFpaSmGDBmS47EmQghx9OhR0bp1a+l4dHNzE0uXLpXm3717V3Tu3Fmo1WphZmYmunXrJu7du5fr75XvvvtOVKxYUejo6Gh8315/3IkQWcfbp59+KtRqtTA0NBQNGzYUu3bt0miT1/H++rF28+ZNMWDAAFG1alVhaGgozM3NRYsWLcSBAwfe2p8fKoUQxXRVNhERkRZ07twZly5dyvEJBsXNz88PW7ZsKbEReqLc8Bo7IiKSjfv372P37t3vfNME0fuK19gREdF7LyYmBhEREVizZg309fXfeD0kkZxxxI6IiN574eHh+PzzzxETE4OQkBBYWVlpuyQireA1dkREREQywRE7IiIiIplgsCMiIiKSCd48QYWSmZmJe/fuoUyZMkX6EUlERESU9akgz549g42NTYEe7MxgR4Vy79492NraarsMIiIiWbtz5w4qVaqU7/YMdlQo2R8ZdefOHZiammq5GiIiInlJTEyEra1tgT+ikcGOCiX79KupqSmDHRERUTEp6OVOvHmCiIiISCYY7IiIiIhkgsGOiIiISCZ4jR0REZHMZGRkIC0tTdtl0FsYGBgU6FEm+cFgR0REJBNCCDx48ADx8fHaLoXyQUdHBw4ODjAwMCiydTLYERERyUR2qKtQoQKMjY35APlSLPtB//fv30flypWL7HvFYEdERCQDGRkZUqgrV66ctsuhfLCwsMC9e/eQnp4OfX39Ilkngx29k1pT90FHaaztMoiISoVYw15a23aaSSXgo4UwNn4J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EREAyO+aMCodOGJHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBMMdkREREQywWBHREREJBN62i6A3m+1pu6DjtJY22UQEZWoWMNe2i6BXhWQoO0KSg2O2BERERHJBIMdERERkUwU6lRsRkYGgoODERoairi4OGRmZmrMP3jwYJEUR0RERET5V6hgN2rUKAQHB6Ndu3aoVasWFApFUddFr7C3t8fo0aMxevRobZdCREREpVihgt3GjRuxefNmtG3btqjrkRU/Pz/Ex8dj+/btGtPDwsLQokULPH36FGq1Wiu1ZcurRiIiInr/FCrYGRgYoFq1akVdC5WgjIwMjrQSERHJTKFunhg7diwWL14MIURR1/NB+v3331GzZk0olUrY29tjwYIFOdo8e/YMPXv2hImJCSpWrIjly5drzF+4cCFcXV1hYmICW1tbDB06FElJSdL84OBgqNVq7Ny5EzVq1IBSqcSAAQMQEhKCHTt2QKFQQKFQICwsrLh3l4iIiIpJoUbsjh49ikOHDmHPnj2oWbMm9PX1NeZv3bq1SIr7EJw5cwbdu3dHQEAAevTogb/++gtDhw5FuXLl4OfnJ7WbN28eJk2ahGnTpmHfvn0YNWoUnJyc0Lp1awCAjo4OlixZAgcHB9y8eRNDhw7FN998gxUrVkjrePHiBebMmYM1a9agXLlysLa2xsuXL5GYmIigoCAAgLm5ea51pqSkICUlRXqfmJhYDL1BRERE76JQwU6tVqNz585FXYss7dq1CyqVSmNaRkaG9PXChQvRqlUrTJ48GQDg5OSEK1euYN68eRrBztPTExMmTJDaRERE4Pvvv5eC3as3Vtjb22PGjBkYPHiwRrBLS0vDihUr4O7uLk0zMjJCSkoKrKys3rgfgYGBmDZtWsF2noiIiEpUoYJd9ugOvV2LFi2wcuVKjWknTpxAnz59AABRUVHo2LGjxnxPT08sWrQIGRkZ0NXVBQB4eHhotPHw8MCiRYuk9wcOHEBgYCD+/vtvJCYmIj09HcnJyXjx4gWMjbM+GcLAwABubm6F2o+JEydizJgx0vvExETY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tW10dDRSU1M16jA3N4ezs7P0/tKlS8jIyICTk5NGHeHh4W+sIzAwEGZmZtLL1tb2Db1FRERE2sARu3xau3Yt0tPTpVEuABBCQKlUYtmyZW9dPikpCQDw448/5giAurq6uS6zceNG+Pv7Y8GCBfDw8ECZMmUwb948nDhxotD7kZSUBF1dXZw5cybHdlUqVZ7LTZw4EWPGjJHeJyYmMtwRERGVMgx2+ZCeno5169ZhwYIF+PjjjzXmderUCRs2bICLiwtOnDiBvn37SvOOHz8ufW1paQkbGxvcvHkTvXv3ztd2IyIi0KRJEwwdOlSa9qZRtapVq0JfXx8nTpxA5cqVAQBPnz7FtWvX0Lx5cwBAnTp1kJGRgbi4ODRt2jRfdQCAUqmEUqnMd3siIiIqeQx2+bBr1y48ffoUAwcOhJmZmca8rl27Yu3atfD394efnx/q168PT09PrF+/HpcvX0aVKlWkttOmTcPIkSNhZmYGX19fpKSk4PTp03j69KnGaFg2R0dHrFu3Dvv27YODgwN+/vlnnDp1Cg4ODrnWqVKpMHDgQIwbNw7lypVDhQoV8O2330JH539n3J2cnNC7d2/07dsXCxYsQJ06dfDo0SOEhobCzc2t1Dyfj4iIiAqO19jlw9q1a+Ht7Z0j1AFZwe706dNwcXHB5MmT8c0336BevXq4desWhgwZotF20KBBWLNmDYKCguDq6ormzZsjODg4z6D21VdfoUuXLujRowcaNWqEx48fa4ze5WbevHlo2rQpOnToAG9vb3z00UeoV6+eRpugoCD07dsXY8eOhbOzMzp16oRTp05Jo3xERET0flKI7Od1EBVAYmJi1k0UozdDR2ms7XKIiEqdWMNe2i7hwxGQoO0Kilz239mEhASYmprmezmO2BERERHJBIMdERERkUww2BERERHJBO+KpXcSOc2nQOf+iYg+HPK77otKP47YEREREckEgx0RERGRTDDYEREREckEgx0RERGRTDDYEREREckEgx0RERGRTDDYEREREckEgx0RERGRTPABxVQoQggAWR9STEREREUr++9r9t/b/GKwo0J5/PgxAMDW1lbLlRAREcnXs2fPYGZmlu/2DHZUKObm5gCA27dvF+iAk5PExETY2trizp07H/THqrEfsrAfsrAfsrAf2AfZCtsPQgg8e/YMNjY2Bdoegx0Vio5O1uWZ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\n",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Select top 10 locations with the highest counts\n",
    "top_locations = df['Location'].value_counts().head(10).index\n",
    "\n",
    "# Filter the DataFrame for top locations\n",
    "top_location_df = df[df['Location'].isin(top_locations)]\n",
    "\n",
    "# Visualize the distribution of RainTomorrow across top locations\n",
    "plt.figure(figsize=(10, 6))\n",
    "location_rain_distribution = top_location_df.groupby(['Location', 'RainTomorrow']).size().unstack()\n",
    "location_rain_distribution.plot(kind='barh', stacked=True)\n",
    "plt.title('RainTomorrow Distribution by Top Locations')\n",
    "plt.xlabel('Count')\n",
    "plt.ylabel('Location')\n",
    "plt.legend(title='RainTomorrow')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3dc99b27",
   "metadata": {},
   "source": [
    "# Training and Testing Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "3bd0556c",
   "metadata": {},
   "outputs": [],
   "source": [
    "features = ['MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed', 'Humidity9am', 'Humidity3pm']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "c7c2d71f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare the data\n",
    "# influence of x on y\n",
    "# y is the target variable should be in a binary classification \n",
    "\n",
    "X = df[features]\n",
    "y = df['RainTomorrow']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "39a23116",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    113583\n",
       "1     31877\n",
       "Name: RainTomorrow, dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Convert the categorical target of 'RainTomorrow' to binary (0 or 1)\n",
    "# The purpose of this operation is to prepare the target variable in a format that's suitable for\n",
    "# binary classification models. \n",
    "df['RainTomorrow'] = df['RainTomorrow'].replace({'Yes': 1, 'No': 0})\n",
    "\n",
    "df['RainTomorrow'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "8027df94",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Lets Split the data into training and testing sets \n",
    "\n",
    "X_train, X_test, y_train, y_test, = train_test_split(X, y, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "4912ddb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standardise the features for a better performance \n",
    "\n",
    "# create a StandardScaler instance \n",
    "scaler = StandardScaler()\n",
    "\n",
    "# Fit the scaler on training data and transform it\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "\n",
    "# Transform the test data using the scaler\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d8bf75e",
   "metadata": {},
   "source": [
    "# Training the Model and Evaluation "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d922e6f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model training and evaluation\n",
    "# Create and train the logistic regression model\n",
    "\n",
    "model = LogisticRegression()\n",
    "model.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Make prediction on test set\n",
    "y_pred = model.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "52bb30a2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.8327031486319263\n"
     ]
    }
   ],
   "source": [
    "# calculating accuracy \n",
    "\n",
    "\n",
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(\"Accuracy:\", accuracy)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "feed4ba8",
   "metadata": {},
   "source": [
    "#### the model predict about 83.27% whether will it rain tomorrow or now using our features "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "7d024f2a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAApoAAAIjCAYAAACjybtCAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8o6BhiAAAACXBIWXMAAA9hAAAPYQGoP6dpAABXvklEQVR4nO3de3zO9f/H8ee107WDHQw7lcNQDjmrtMqplpGU6IDKIZIaxYTWt7T41ookIupXoqLQQYVoiCUTxpyKr2P7Fpuz2bDj9fvDd1ddzWFbe7dxPe632+d2c70/78/78/58bm179XofLovNZrMJAAAAKGMu5d0BAAAAXJkINAEAAGAEgSYAAACMINAEAACAEQSaAAAAMIJAEwAAAEYQaAIAAMAIAk0AAAAYQaAJAAAAIwg0AVzUrl271KFDB/n7+8tisWjBggVl2v7+/ftlsVg0c+bMMm33ctauXTu1a9euvLsBAH8bgSZwGdizZ48ef/xx1a5dW56envLz89Mtt9yiSZMm6cyZM0bv3adPH23dulUvv/yyPvroI11//fVG7/dP6tu3rywWi/z8/M77Hnft2iWLxSKLxaLXX3+9xO0fOHBAcXFxSklJKYPeAsDlx628OwDg4hYtWqT7779fVqtVvXv3VqNGjZSTk6PVq1drxIgR2r59u959910j9z5z5oySkpL0r3/9S4MHDzZyj5o1a+rMmTNyd3c30v6luLm56fTp0/rmm2/0wAMPOJybPXu2PD09dfbs2VK1feDAAb300kuqVauWmjVrVuzrvvvuu1LdDwAqGgJNoALbt2+fevTooZo1a2rFihUKDQ21n4uOjtbu3bu1aNEiY/c/fPiwJCkgIMDYPSwWizw9PY21fylWq1W33HKLPvnkkyKB5pw5c9S5c2d9/vnn/0hfTp8+LW9vb3l4ePwj9wMA0xg6ByqwcePGKTMzU++//75DkFmobt26evrpp+2f8/LyNHbsWNWpU0dWq1W1atXSc889p+zsbIfratWqpbvuukurV6/WjTfeKE9PT9WuXVsffvihvU5cXJxq1qwpSRoxYoQsFotq1aol6dyQc+G//ywuLk4Wi8WhLCEhQbfeeqsCAgJUqVIl1atXT88995z9/IXmaK5YsUKtW7eWj4+PAgICdM899+iXX3457/12796tvn37KiAgQP7+/urXr59Onz594Rf7F7169dK3336rEydO2MvWr1+vXbt2qVevXkXqHzt2TM8884waN26sSpUqyc/PT506ddLmzZvtdVauXKkbbrhBktSvXz/7EHzhc7Zr106NGjVScnKy2rRpI29vb/t7+esczT59+sjT07PI80dFRaly5co6cOBAsZ8VAP5JBJpABfbNN9+odu3auvnmm4tVf8CAARo9erRatGihiRMnqm3btoqPj1ePHj2K1N29e7fuu+8+3XHHHZowYYIqV66svn37avv27ZKkbt26aeLEiZKknj176qOPPtKbb75Zov5v375dd911l7KzszVmzBhNmDBBd999t3788ceLXrds2TJFRUXp0KFDiouLU0xMjNasWaNbbrlF+/fvL1L/gQce0KlTpxQfH68HHnhAM2fO1EsvvVTsfnbr1k0Wi0VffPGFvWzOnDmqX7++WrRoUaT+3r17tWDBAt1111164403NGLECG3dulVt27a1B30NGjTQmDFjJEkDBw7URx99pI8++kht2rSxt3P06FF16tRJzZo105tvvqn27duft3+TJk1StWrV1KdPH+Xn50uS3nnnHX333Xd66623FBYWVuxnBYB/lA1AhXTy5EmbJNs999xTrPopKSk2SbYBAwY4lD/zzDM2SbYVK1bYy2rWrGmTZEtMTLSXHTp0yGa1Wm3Dhw+3l+3bt88myTZ+/HiHNvv06WOrWbNmkT68+OKLtj//Wpk4caJNku3w4cMX7HfhPT744AN7WbNmzWxBQUG2o0eP2ss2b95sc3FxsfXu3bvI/R599FGHNu+9915blSpVLnjPPz+Hj4+PzWaz2e677z7b7bffbrPZbLb8/HxbSEiI7aWXXjrvOzh79qwtPz+/yHNYrVbbmDFj7GXr168v8myF2rZta5Nkmz59+nnPtW3b1qFs6dKlNkm2f//737a9e/faKlWqZOvateslnxEAyhMZTaCCysjIkCT5+voWq/7ixYslSTExMQ7lw4cPl6QiczkbNmyo1q1b2z9Xq1ZN9erV0969e0vd578qnNv51VdfqaCgoFjXHDx4UCkpKerbt68CAwPt5U2aNNEdd9xhf84/GzRokMPn1q1b6+jRo/Z3WBy9evXSypUrlZaWphUrVigtLe28w+bSuXmdLi7nfn3m5+fr6NGj9mkBGzduLPY9rVar+vXrV6y6HTp00OOPP64xY8aoW7du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\n",
      "text/plain": [
       "<Figure size 800x600 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Confusion Matrix - this looks for false/true positives and negatives - ranging from -1 to 1.\n",
    "# [[True Negative (TN)  False Positive (FP)]\n",
    "# [False Negative (FN) True Positive (TP)]]\n",
    "\n",
    "conf_matrix = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(conf_matrix, annot=True,cmap=\"coolwarm\")\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "704e941c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "     No Rain       0.85      0.95      0.90     22672\n",
      "        Rain       0.71      0.41      0.52      6420\n",
      "\n",
      "    accuracy                           0.83     29092\n",
      "   macro avg       0.78      0.68      0.71     29092\n",
      "weighted avg       0.82      0.83      0.82     29092\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Model Evaluation with Precision, Recall, and F1-Score\n",
    "report = classification_report(y_test, y_pred, target_names=['No Rain', 'Rain'])\n",
    "print(\"Classification Report:\\n\", report)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29d5aa3b",
   "metadata": {},
   "source": [
    "### Precision is the percentage of predicted positives that are actually positive. Recall is the percentage of actual positives that are predicted positive. F1-score is a weighted harmonic mean of precision and recall.\n",
    "\n",
    "The overall precision, recall, and F1-score are 0.90, 0.83, and 0.82, respectively. This means that the model is able to correctly predict the class of 90% of the data points, and it is able to identify 83% of the positive data points.\n",
    "\n",
    "The table also shows that the precision for the No Rain class is higher than the precision for the Rain class. This means that the model is more likely to correctly predict that there will be no rain than it is to correctly predict that it will rain.\n",
    "\n",
    "The recall for the Rain class is lower than the recall for the No Rain class. This means that the model is more likely to miss a case of rain than it is to miss a case of no rain.\n",
    "\n",
    "Overall, the classification report shows that the model is performing well. However, there is still room for improvement, especially in the recall for the Rain class."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "41aa0343",
   "metadata": {},
   "source": [
    "## will it rain tomorrow ? "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "6096f8aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "No, it will not rain tomorrow.\n"
     ]
    }
   ],
   "source": [
    "# Define the new input features as a dictionary for conditions of a given day\n",
    "new_input = {'MinTemp': 17.8, 'MaxTemp': 22, 'Rainfall': 19,\n",
    "             'WindGustSpeed': 24, 'Humidity9am': 75, 'Humidity3pm': 85}\n",
    "\n",
    "# Create a NumPy array from the dictionary values\n",
    "new_input_np = np.array(list(new_input.values())).reshape(1, -1)\n",
    "new_input_np\n",
    "\n",
    "# Make predictions on the new input\n",
    "prediction = model.predict(new_input_np)\n",
    "\n",
    "if prediction == 1:\n",
    "    print('Yes, it will rain tomorrow.')\n",
    "else:\n",
    "    print('No, it will not rain tomorrow.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fc189291",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
