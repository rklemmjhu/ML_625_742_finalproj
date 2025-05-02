### Imports
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

### Load in data
movie_per_service = pd.read_csv('./MoviesOnStreamingPlatforms_updated.csv')
boxoffice_gross_data = pd.read_csv('./boxoffice_gross_data.csv')

### Preparation of training datasets

data = movie_per_service.join(other=boxoffice_gross_data.rename(columns={'title': 'Title'}).set_index('Title'), on='Title', lsuffix='_L', rsuffix='_R')

# Build first dataframe

# Type column is always 0, studio and director would add too many dimensions
# rank is not useful for us
# Age is missing in too many rows to form a good dataset
# lowercase year is a duplicate
data = data.drop(columns=['Type', 'studio', 'Directors', 'rank', 'year', 'Age'])

# Recover age column
#data['Age'] = data['Age'].apply(lambda x: str(x).replace('+','') if type(x) is str else 0 if np.isnan(x) else int(x))

# Trim rows that are missing a country
data = data[data['Country'].notna() & (data['Country'].str.strip() != '')].copy()

# Trim rows that are missing a lanuage
data = data[data['Language'].notna() & (data['Language'].str.strip() != '')].copy()

# Trim rows that are missing a genre
data = data[data['Genres'].notna() & (data['Genres'].str.strip() != '')].copy()

# Trim rows that are missing an IMDb score
data = data[data['IMDb'].notna()].copy()

# Trim rows that are missing a runtime
data = data[data['Runtime'].notna()].copy()

# Netflix defaults to float for some reason, bring it back to int
data['Netflix'] = data['Netflix'].astype(int)

# Generate one-hot genre encoding
data['genres_list'] = data['Genres'].str.split(',')
genre_mlb = MultiLabelBinarizer()
genre_ = pd.DataFrame(genre_mlb.fit_transform(data['genres_list']), columns=genre_mlb.classes_, index=data.index)
genre_list = genre_mlb.classes_
data = pd.concat([data.drop(columns=['Genres', 'genres_list']), genre_], axis=1)

# Generate one-hot langauge encoding
data['languages_list'] = data['Language'].str.split(',')
language_mlb = MultiLabelBinarizer()
language_ = pd.DataFrame(language_mlb.fit_transform(data['languages_list']), columns=language_mlb.classes_, index=data.index)
language_list = language_mlb.classes_
data = pd.concat([data.drop(columns=['Language', 'languages_list']), language_], axis=1)

# Generate one-hot country encoding
data['country_list'] = data['Country'].str.split(',')
country_mlb = MultiLabelBinarizer()
country_ = pd.DataFrame(country_mlb.fit_transform(data['country_list']), columns=country_mlb.classes_, index=data.index)
country_list = country_mlb.classes_
data = pd.concat([data.drop(columns=['Country', 'country_list']), country_], axis=1)

# Recover streaming service class from one-hot encoding
service_columns = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']
data['Service'] = data[service_columns].apply(
    lambda row: ','.join(col for col in service_columns if row[col] == 1),
    axis=1
)
data = data.drop(columns=service_columns)

df_full = data.copy()

# Build second dataframe

# Trim rows missing a gross amount
data = data[data['lifetime_gross'].notna()].copy()

# Trim rows missing a Rotten Tomatoes Score
data = data[data['Rotten Tomatoes'].notna()].copy()

# Bring rotten tomatoes score to a number
data['Rotten Tomatoes'] = data['Rotten Tomatoes'].str.rstrip('%').astype(float)

df_boxoffice = data.copy()

print("Data Preview:")
print(data.head())

### Train various types of models on a given dataframe

# Both algorithms require data normalization. This function
# returns the X table for training
def preprocess(df, input_cols_num, input_cols_cat):
  num_df = df[input_cols_num].copy()
  cat_df = df[input_cols_cat].copy()

  scaler = StandardScaler()
  scaled_num = scaler.fit_transform(num_df)

  scaled_num_df = pd.DataFrame(scaled_num, columns=input_cols_num, index=df.index)


  return pd.concat([scaled_num_df, cat_df], axis=1)


def train_knn(df, input_cols_num, input_cols_cat, output_col, k, feature_summ):
  X = preprocess(df, input_cols_num, input_cols_cat)
  y = df[output_col].copy()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  model = KNeighborsClassifier(n_neighbors=k)
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)
  
  accuracy = accuracy_score(y_test, y_pred)
  print(f"KNN: Features={feature_summ}, k={k}, accuracy={accuracy:.2%}")
  return

def train_svm(df, input_cols_num, input_cols_cat, output_col, C, feature_summ):
  X = preprocess(df, input_cols_num, input_cols_cat)
  y = df[output_col].copy()

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  model = SVC(kernel='rbf', C=C, gamma='scale')
  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  print(f"SVM: Features={feature_summ}, C={C}, accuracy={accuracy:.2%}")
  return


print()
print()

### Test models on the full dataset, large N
print(f"Large dataset test, N={len(df_full)}")
numerical_features = ["Year", "IMDb", "Runtime"]
output_col = "Service"

# Try multiple different hyperparameters
k_list = [1, 5, 10, 15, 20]
for k in k_list:
  train_knn(df_full, numerical_features, [], output_col, k, "Numerical Only")

  train_knn(df_full, numerical_features, genre_list, output_col, k, "Numerical + Genre")

  train_knn(df_full, numerical_features, np.concatenate((genre_list, language_list, country_list)), output_col, k, "Numerical + Genre + Language / Country")

C_list = [0.01, 0.1, 1, 10, 100]
for C in C_list:
  train_svm(df_full, numerical_features, [], output_col, C, "Numerical Only")

  train_svm(df_full, numerical_features, genre_list, output_col, C, "Numerical + Genre")

  train_svm(df_full, numerical_features, np.concatenate((genre_list, language_list, country_list)), output_col, C, "Numerical + Genre + Language / Country")
  
  
print()
print()

### Test models on the dataset including box office data, small N
print(f"Smaller dataset test, N={len(df_boxoffice)}")
numerical_features = ["Year", "IMDb", "Runtime", "lifetime_gross", "Rotten Tomatoes"]
output_col = "Service"

# Try multiple different hyperparameters
k_list = [1, 5, 10, 15, 20]
for k in k_list:
  train_knn(df_boxoffice, numerical_features, [], output_col, k, "Numerical Only")

  train_knn(df_boxoffice, numerical_features, genre_list, output_col, k, "Numerical + Genre")

  train_knn(df_boxoffice, numerical_features, np.concatenate((genre_list, language_list, country_list)), output_col, k, "Numerical + Genre + Language / Country")

C_list = [0.01, 0.1, 1, 10, 100]
for C in C_list:
  train_svm(df_boxoffice, numerical_features, [], output_col, C, "Numerical Only")

  train_svm(df_boxoffice, numerical_features, genre_list, output_col, C, "Numerical + Genre")

  train_svm(df_boxoffice, numerical_features, np.concatenate((genre_list, language_list, country_list)), output_col, C, "Numerical + Genre + Language / Country")