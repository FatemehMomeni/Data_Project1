from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# showing data frames completely
pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', 61)
pd.set_option('display.max_rows', 65000)

# reading csv files
questions = pd.read_csv('F:/Mine/terms/masters/term2/dataMining/hw/HW1 (Chapters 1-3)/Questions.csv')
answers = pd.read_csv('F:/Mine/terms/masters/term2/dataMining/hw/HW1 (Chapters 1-3)/Answers.csv')

# counting number of missing values per column
NaN_num = answers.isna().sum()

# a lis for storing percentage of missing values in each column
NaN_percent = dict()
answers_columns = answers.columns
total_num = len(answers)
# calculating percentage of missing values per column
for num in range(len(NaN_num)):
    NaN_percent[answers_columns[num]] = NaN_num[num] * 100 / total_num

# sorting list of percentages to find columns with max number of missing values that stored in max_miss list
sort = sorted(NaN_percent.items(), key=lambda x: x[1], reverse=True)
max_miss = list()
for i in sort:
    if i[1] >= 70.0:
        max_miss.append(i[0])
    else:
        break

# deleting columns have many missing values
processed_ans = answers.drop(axis=1, columns=max_miss)

# finding type of columns
# print(processed_ans.dtypes)

# filling missing values with mean for numeric(float64) columns
numeric = ['Age', 'CompTotal', 'ConvertedComp', 'WorkWeekHrs']
for i in numeric:
    processed_ans[i] = processed_ans[i].fillna(processed_ans[i].mean())

# filling missing values with mode for categorical columns
categorical = ['MainBranch', 'Hobbyist', 'Age1stCode', 'CompFreq', 'Country', 'CurrencyDesc', 'CurrencySymbol',
               'EdLevel', 'Employment', 'Ethnicity', 'Gender', 'JobSat', 'JobSeek', 'NEWDevOps', 'NEWDevOpsImpt',
               'NEWEdImpt', 'NEWLearn', 'NEWOffTopic', 'NEWOnboardGood', 'NEWOtherComms', 'NEWOvertime',
               'NEWPurchaseResearch', 'NEWPurpleLink', 'OpSys', 'OrgSize', 'PurchaseWhat', 'Sexuality', 'SOAccount',
               'SOComm', 'SOPartFreq', 'SOVisitFreq', 'SurveyEase', 'SurveyLength', 'Trans', 'UndergradMajor',
               'WelcomeChange', 'YearsCode', 'YearsCodePro']
for i in categorical:
    processed_ans[i] = processed_ans[i].fillna(processed_ans[i].mode()[0])

# remained columns
remained = list()
for i in answers_columns:
    if i not in (max_miss + numeric + categorical):
        remained.append(i)

multi_value = dict()
for i in range(1, len(remained)):
    multi_value.clear()
    column = processed_ans[remained[i]]
    for row in column:
        values = str(row).split(';')
        for val in values:
            if val != 'nan':
                if val not in multi_value.keys():
                    multi_value[val] = 1
                else:
                    multi_value[val] += 1
    fill_value = max(multi_value, key=multi_value.get)
    processed_ans[remained[i]] = processed_ans[remained[i]].fillna(fill_value)

# use label encoder to encode object data types
encod_ans = processed_ans.copy()
object_type = categorical + remained
label_encoder = preprocessing.LabelEncoder()
for i in object_type:
    encod_ans[i] = label_encoder.fit_transform(encod_ans[i])

# show boxplot of all columns
"""for i in encod_ans.columns:
    plt.boxplot(encod_ans[i])
    plt.title(i)
    plt.show()"""

# features without outlier
no_outlier_columns = ['Respondent', 'CompFreq', 'Country', 'CurrencyDesc', 'CurrencySymbol', 'JobFactors',
                      'JobSat', 'JobSeek', 'LanguageDesireNextYear', 'LanguageWorkedWith', 'MiscTechWorkedWith',
                      'NEWCollabToolsWorkedWith', 'NEWDevOps', 'NEWEdImpt', 'NEWJobHuntResearch', 'NEWLearn',
                      'NEWOffTopic', 'NEWOnboardGood', 'NEWOtherComms', 'NEWOvertime', 'NEWPurchaseResearch',
                      'NEWPurpleLink', 'NEWStuck', 'OrgSize', 'PlatformDesireNextYear', 'PlatformWorkedWith', 'SOComm',
                      'SOVisitFreq', 'SurveyEase', 'YearsCode', 'YearsCodePro']
# discrete columns (no outlier)
discrete_columns = ['Hobbyist', 'CompTotal', 'Gender', 'PurchaseWhat', 'Sexuality', 'SOAccount', 'SurveyLength', 'Trans',
                    'UndergradMajor', 'WelcomeChange']

# features with outlier
outlier_columns = list()
for i in encod_ans.columns:
    if i not in (no_outlier_columns + discrete_columns):
        outlier_columns.append(i)

# detecting outliers **** i think it isn't needed. Because we can use boxplot instead
"""for i in outlier_columns:
    plt.plot(encod_ans[i], 'o')
    plt.title(i)
    plt.show()"""

# columns which their outliers should be removed
removes = ['MainBranch', 'EdLevel', 'Employment', 'NEWDevOpsImpt', 'OpSys', 'SOPartFreq']

# outlier removal
removable = list()
for column in removes:
    encod_ans_sort = encod_ans.sort_values(by=column, ascending=True)
    q1 = np.quantile(encod_ans_sort[column], 0.25)
    q3 = np.quantile(encod_ans_sort[column], 0.75)
    IQR = q3 - q1
    lower_fence = q1 - (1.5 * IQR)
    upper_fence = q3 + (1.5 * IQR)
    removable.clear()
    for row in range(len(encod_ans)):
        if encod_ans.loc[row, column] < lower_fence or encod_ans.loc[row, column] > upper_fence:
            removable.append(row)
    encod_ans.drop(removable, inplace=True)
    encod_ans = encod_ans.reset_index(drop=True)

# imputing outlier with median
outliers = list()
for column in outlier_columns:
    encod_ans_sort = encod_ans.sort_values(by=column, ascending=True)
    q1 = np.quantile(encod_ans_sort[column], 0.25)
    q3 = np.quantile(encod_ans_sort[column], 0.75)
    IQR = q3 - q1
    lower_fence = q1 - (1.5 * IQR)
    upper_fence = q3 + (1.5 * IQR)
    outliers.clear()
    for row in range(len(encod_ans[column])):
        if encod_ans.loc[row, column] < lower_fence or encod_ans.loc[row, column] > upper_fence:
            outliers.append(row)
    median = np.median(encod_ans[column])
    for out in outliers:
        encod_ans.loc[out, column] = median

# columns dependency
"""sns.scatterplot(x="Age", y="EdLevel", data=encod_ans)
plt.show()
sns.scatterplot(x="Age1stCode", y="YearsCode", data=encod_ans)
plt.show()
sns.scatterplot(x="LanguageWorkedWith", y="LanguageDesireNextYear", data=encod_ans)
plt.show()"""

# distribution
"""sns.displot(encod_ans['Age'])
plt.show()
sns.kdeplot(encod_ans['Age'])
plt.show()
sns.displot(encod_ans['EdLevel'])
plt.show()
sns.kdeplot(encod_ans['EdLevel'])
plt.show()
sns.displot(encod_ans['Gender'])
plt.show()
sns.kdeplot(encod_ans['Gender'])
plt.show()"""

# print(encod_ans.isnull().values.any())
# print(np.all(np.isfinite(encod_ans)))
encod_ans = pd.DataFrame(np.nan_to_num(encod_ans), columns=answers_columns)

# PCA
pca_features = ['MainBranch', 'Hobbyist', 'Age', 'Age1stCode', 'CompFreq', 'CompTotal', 'ConvertedComp', 'Country']
other_features = list()
for i in encod_ans.columns:
    if i not in pca_features:
        other_features.append(i)

pca_ans = encod_ans.loc[:, pca_features].values
pca_ans = StandardScaler().fit_transform(pca_ans)
pca = PCA(n_components=2)
pc = pca.fit_transform(pd.DataFrame(np.nan_to_num(pca_ans), columns=pca_features))
pc_df = pd.DataFrame(data=pc, columns=['principal component 1', 'principal component 2'])
final_ans = pd.concat([pc_df, encod_ans[other_features]], axis=1)

# LanguageWorkedWith
languages = list()
for row in processed_ans['LanguageWorkedWith']:
    values = str(row).split(';')
    for v in values:
        if v not in languages:
            languages.append(v)

lww_df = pd.DataFrame(0, index=np.arange(len(processed_ans)), columns=languages)
for row in range(len(processed_ans)):
    values = str(processed_ans.loc[row, 'LanguageWorkedWith']).split(';')
    for e in values:
        lww_df.loc[row, e] = 1
    values.clear()

# most useful programming languages
summation = dict()
for i in lww_df.columns:
    summation[i] = lww_df[i].sum()
max_sum = max(summation.values())
for i in summation:
    if summation[i] == max_sum:
        print(i)

# countries with max salary
df = encod_ans.copy()
df['Avg'] = pd.DataFrame(df.groupby('Country')['CompTotal'].agg(Avg='mean')).reset_index()['Avg']
print(max(df['Avg']))

# countries with max work hours
df2 = encod_ans.copy()
df2['Avg'] = pd.DataFrame(df.groupby('Country')['WorkWeekHrs'].agg(Avg='mean')).reset_index()['Avg']
print(max(df2['Avg']))

# word cloud
wordcloud = WordCloud(background_color="white").generate(str(encod_ans['Country']))
plt.figure(figsize=(20, 20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
