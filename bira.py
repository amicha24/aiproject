import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as go

sns.set(style="whitegrid")

beer_recipe = pd.read_csv('recipeData.csv', index_col='BeerID', encoding='latin1')
beer_recipe.head()

print('Number of recipes =\t\t{} \nNumber of beer styles =\t{}'.format(len(beer_recipe), len(beer_recipe['Style'].unique())))


broad_styles = ['Ale', 'IPA', 'Pale Ale', 'Lager', 'Stout', 'Bitter', 'Cider', 'Porter']
beer_recipe['BroadStyle'] = 'Other'
beer_recipe['Style'].fillna('Unknown', inplace=True)
for broad_style in broad_styles:
    beer_recipe.loc[beer_recipe['Style'].str.contains(broad_style), 'BroadStyle'] = broad_style
style_popularity_as_perc = 100 * beer_recipe['BroadStyle'].value_counts() / len(beer_recipe)
style_popularity_as_perc.drop('Other', inplace=True)

pltly_data = [go.Bar(x=style_popularity_as_perc.index,
                     y=style_popularity_as_perc.values)]

layout = go.Layout(title='Most popular general styles',
                   xaxis={'title': 'Style'},
                   yaxis={'title': 'Proportion of recipes (%)'})

fig = go.Figure(data=pltly_data, layout=layout)
py.iplot(fig)



fig, ax = plt.subplots(1, 1, figsize=[12,5])
sns.distplot(beer_recipe['ABV'], ax=ax)
ax.set_title('ABV distribution')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()

strengths = [10, 15, 20, 30, 50]
for abv in strengths:
    print('{} ({:.2f})%\tbeers stronger than {} ABV'.format(sum(beer_recipe['ABV'] > abv), 100 * sum(beer_recipe['ABV'] > abv) / len(beer_recipe), abv))

abv_df = beer_recipe[beer_recipe['ABV'] <= 15]

fig, ax = plt.subplots(1, 1, figsize=[12, 5])
sns.violinplot(x='BroadStyle',
               y='ABV',
               data=abv_df,
               ax=ax)
ax.set_xlabel('General beer style')
ax.set_title('ABV by beer style')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


beer_recipe['gravity_change'] = beer_recipe['OG'] - beer_recipe['FG']  # Suspect this may be correlated with ABV

df_for_corr = beer_recipe.drop(['Style', 'BroadStyle', 'StyleID'], axis=1).copy()
categoricals = df_for_corr.columns[df_for_corr.dtypes == 'object']
for categorical in categoricals:
    print('{} has {} unique values'.format(categorical, len(df_for_corr[categorical].unique())))
    if len(df_for_corr[categorical].unique()) > 20:
           df_for_corr.drop(categorical, axis=1, inplace=True)


encoded_df = pd.get_dummies(df_for_corr)
corr_mat = encoded_df.corr()
abv_corrs = corr_mat['ABV'].sort_values()
abv_corrs.drop(['ABV', 'Color', 'IBU'], inplace=True)  # Color and IBU are results rather than parts of the brewing process so drop.

fig, ax = plt.subplots(1, 1, figsize=[10, 7])
sns.heatmap(corr_mat.mask(np.triu(np.ones(corr_mat.shape)).astype(np.bool)), ax=ax, center=0)
plt.show()


f,ax = plt.subplots(figsize=(18, 18))
beer_recipe["SugarScale_map"] = beer_recipe["SugarScale"].map({"Specific Gravity":1,"Plato":2}).astype(int)
beer_recipe["BrewMethod_map"] = beer_recipe["BrewMethod"].map({"All Grain":1,"extract":2,"BIAB":3,"Partial Mash":4}).astype(int)
sns.heatmap(beer_recipe.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.savefig("Heatmap.png")
plt.show()

def get_sg_from_plato(plato):
    sg = 1 + (plato / (258.6 - ( (plato/258.2) *227.1) ) )
    return sg

beer_recipe['OG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['OG']) if row['SugarScale'] == 'Plato' else row['OG'], axis=1)
beer_recipe['FG_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['FG']) if row['SugarScale'] == 'Plato' else row['FG'], axis=1)
beer_recipe['BoilGravity_sg'] = beer_recipe.apply(lambda row: get_sg_from_plato(row['BoilGravity']) if row['SugarScale'] == 'Plato' else row['BoilGravity'], axis=1)

# I should define a function that will categorize the features automatically
vlow_scale_feats = ['OG_sg', 'FG_sg', 'BoilGravity_sg', 'PitchRate']
low_scale_feats = ['ABV', 'MashThickness']
mid_scale_feats = ['Color', 'BoilTime', 'Efficiency', 'PrimaryTemp']
high_scale_feats = ['IBU', 'Size(L)',  'BoilSize']


style_cnt = beer_recipe.loc[:,['Style','PrimingMethod']]
style_cnt['NullPriming'] = style_cnt['PrimingMethod'].isnull()
style_cnt['Count'] = 1
style_cnt_grp = style_cnt.loc[:,['Style','Count','NullPriming']].groupby('Style').sum()

style_cnt_grp = style_cnt_grp.sort_values('NullPriming', ascending=False)
style_cnt_grp.reset_index(inplace=True)



data = beer_recipe[['OG','FG','ABV','IBU','Color']]

#Calculate the ABV from the OG and FG and compare it to what the user entered
# Remove recipes where the difference is large, on the assumption they could have entered other things wrong too
data = data.assign(calculatedABV=(data.OG - data.FG) * 131.25)
data = data.assign(abvDiff=np.abs(data.ABV - data.calculatedABV))
print('Removing %d beers where ABV didn\'t match calculated ABV' % data.abvDiff[data.abvDiff >= 1].count())
data = data[data.abvDiff < 0.5]

#Get rid of beer with an ABV less than 2%
print('Removing %d beers with low ABV\n' % data.ABV[data.ABV < 2].count())
data = data[data.ABV >= 2]
data = data.drop(columns=['calculatedABV', 'abvDiff'])

# The Brewer's Friend website and forums suggest that an OG of 1.07 is considered high
# See what the OG of the beer with the highest ABV is
max_abv = data.ABV.max()
max_beer = data[data.ABV == max_abv]
print('Beer with highest ABV:\n')
print(data[data.ABV == max_abv])

# See if there are any beers with an OG higher than that one's and remove them
max_og = max_beer.OG.values[0]
cnt = data.OG[data.OG > max_og].count()
print('\n%d rows with unusually high OG' % cnt)
if cnt > 0:
    data = data[data.OG <= max_og]


# Shuffle the data
data = data.sample(frac=1)

# Training/CV/Test split will be 70/15/15
cnt = data.shape[0]
index1 = int(np.ceil(cnt * 0.7))
index2 = int(np.ceil(cnt * 0.85))
data_train = data.iloc[1:index1]
data_cv = data.iloc[index1+1:index2]
data_test = data.iloc[index2+1:]
print('Sizes:\nTraining set - %d\nCV set       - %d\nTest set     - %d' % (beer_recipe.shape[0], data_cv.shape[0], beer_recipe.shape[0]))

mu = data_train.mean()
s = data_train.std()

def normalizeData(d):
    return (d - mu) / s

train_norm = normalizeData(data_train)
cv_norm = normalizeData(data_cv)
test_norm = normalizeData(data_test)

from sklearn.cluster import KMeans

# Calculate distortion the old fashioned way, unoptimized
def calculate_distortion(model, X):
    centroids = model.cluster_centers_
    indices = model.predict(X)
    m = X.shape[0]

    J = 0
    for i in range(0,m):
        c = centroids[indices[i]]
        J += (X.iloc[i] - c).sum() ** 2
    return J/m

# There are 176 unique styles in the original data
cluster_counts = np.arange(10,180,10)
models = [None] * len(cluster_counts)
costs = [None] * len(cluster_counts)

# This is a long process because I don't know much about optimization
for i in np.arange(0,len(cluster_counts)):
    models[i] = KMeans(n_clusters = cluster_counts[i]).fit(train_norm)
    costs[i] = calculate_distortion(models[i], cv_norm)

#Graph the data
plt.figure(figsize=(10,6))
plt.plot(cluster_counts, costs, 'o')
plt.xlabel('Clusters ( K )')
plt.ylabel('Distortion ( J )')
plt.show()


def stacked_bar_plot(df, x_total, x_sub_total, sub_total_label, y):
    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(12, 8))

    # Plot the total
    sns.set_color_codes("pastel")
    sns.barplot(x=x_total, y=y, data=df, label="Total", color="b")

    # Plot
    sns.set_color_codes("muted")
    sns.barplot(x=x_sub_total, y=y, data=df, label=sub_total_label, color="b")

    # Add a legend and informative axis label
    ax.legend(ncol=2, loc="lower right", frameon=True)
    sns.despine(left=True, bottom=True)

    return f, ax

f, ax = stacked_bar_plot(style_cnt_grp[:20], 'Count', 'NullPriming', 'Priming Method is null', 'Style')
ax.set(title='Missing Values in PrimingMethod column, per style', ylabel='', xlabel='Count of Beer Recipes')
sns.despine(left=True, bottom=True)


num_feats_list = ['Size(L)', 'OG_sg', 'FG_sg', 'ABV', 'IBU', 'Color', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'PitchRate', 'PrimaryTemp']
beer_recipe.loc[:, num_feats_list].describe().T

# Get top10 styles
top10_style = list(style_cnt_grp['Style'][:10].values)

# Group by current count information computed earlier and group every style not in top20 together
style_cnt_other = style_cnt_grp.loc[:, ['Style','Count']]
style_cnt_other.Style = style_cnt_grp.Style.apply(lambda x: x if x in top10_style else 'Other')
style_cnt_other = style_cnt_other.groupby('Style').sum()

# Get ratio of each style
style_cnt_other['Ratio'] = style_cnt_other.Count.apply(lambda x: x/float(len(beer_recipe)))
style_cnt_other = style_cnt_other.sort_values('Count', ascending=False)

f, ax = plt.subplots(figsize=(8, 8))
explode = (0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0)
plt.pie(x=style_cnt_other['Ratio'], labels=list(style_cnt_other.index), startangle = 180, autopct='%1.1f%%', pctdistance= .9, explode=explode)
plt.title('Ratio of styles across dataset')
plt.show()

#plt.barh(list(style_cnt_other.index), style_cnt_other['Count'])
style_cnt_other['Ratio'].plot(kind='barh', figsize=(12,6),)
plt.title('Ratio of styles across dataset')
sns.despine(left=True, bottom=True)
plt.gca().invert_yaxis()

# create specific df that only contains the fields we're interested in
pairplot_df = beer_recipe.loc[:, ['Style','OG_sg','FG_sg','ABV','IBU','Color']]

# create the pairplot
sns.set(style="dark")
sns.pairplot(data=pairplot_df)
plt.show()


style_cnt_grp = style_cnt_grp.sort_values('Count', ascending=False)
top5_style = list(style_cnt_grp['Style'][:5].values)

top5_style_df = pairplot_df[pairplot_df['Style'].isin(top5_style)]

f, ax = plt.subplots(figsize=(12, 8))
sns.violinplot(x='Style', y='OG_sg',data=top5_style_df)
plt.show()

# Get Top5 styles
top5_style = list(style_cnt_grp['Style'][:5].values)
beer_recipe['Top5_Style'] = beer_recipe.Style.apply(lambda x: x if x in top5_style else 'Other')

# Create Reg plot
sns.lmplot(x='ABV', y='OG', hue='Top5_Style', col='Top5_Style', col_wrap=3, data=beer_recipe, n_boot=100)


from sklearn.impute import SimpleImputer

# imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Get only the features to be used from original dataset
features_list= ['StyleID', #target
                'OG_sg','FG_sg','ABV','IBU','Color', #standardized fields
                'SugarScale', 'BrewMethod', #categorical features
                'Size(L)', 'BoilSize', 'BoilTime', 'BoilGravity_sg', 'Efficiency', 'MashThickness', 'PitchRate', 'PrimaryTemp' # other numerical features
                ]

clf_data = beer_recipe.loc[:, features_list]

# Label encoding
cat_feats_to_use = list(clf_data.select_dtypes(include=object).columns)
for feat in cat_feats_to_use:
    encoder = LabelEncoder()
    clf_data[feat] = encoder.fit_transform(clf_data[feat])

# Fill null values
num_feats_to_use = list(clf_data.select_dtypes(exclude=object).columns)
for feat in num_feats_to_use:
    imputer = SimpleImputer(strategy='median')
    clf_data[feat] = imputer.fit_transform(clf_data[feat].values.reshape(-1,1))

# Seperate Targets from Features
X = clf_data.iloc[:, 1:]
y = clf_data.iloc[:, 0] #the target were the first column I included

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=35)

#sanity check making sure everything is in number format and no null values
X.info()

# imports
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#sanity check again
sanity_df = pd.DataFrame(X_train, columns = X.columns)
sanity_df.describe().T
#imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

clf = RandomForestClassifier()
#clf = LogisticRegression()
#clf = xgb.XGBClassifier()
clf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print('Accuracy: {}'.format(score))

feats_imp = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['FeatureImportance'])
feats_imp = feats_imp.sort_values('FeatureImportance', ascending=False)

feats_imp.plot(kind='barh', figsize=(12,6), legend=False)
plt.title('Feature Importance from RandomForest Classifier')
sns.despine(left=True, bottom=True)
plt.gca().invert_yaxis()
