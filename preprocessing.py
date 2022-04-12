from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# print(answers.loc[3, 'Age'])
# encod_ans.plot(kind='box', figsize=(15, 17))
# plt.boxplot(encod_ans['MainBranch'])

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
    if i[1] >= 40.0:
        max_miss.append(i[0])
    else:
        break

# deleting columns have many missing values
processed_ans = answers.drop(axis=1, columns=max_miss)

# finding type of columns
# print(processed_ans.dtypes)

# filling missing values with mean for numeric(float64) columns
numeric = ['Age', 'WorkWeekHrs']
for i in numeric:
    processed_ans[i] = processed_ans[i].fillna(processed_ans[i].mean())

# filling missing values with mode for categorical columns
categorical = ['MainBranch', 'Hobbyist', 'Age1stCode', 'CompFreq', 'Country', 'CurrencyDesc', 'CurrencySymbol',
               'EdLevel', 'Employment', 'Ethnicity', 'Gender', 'JobSat', 'JobSeek', 'NEWDevOps', 'NEWDevOpsImpt',
               'NEWEdImpt', 'NEWLearn', 'NEWOffTopic', 'NEWOnboardGood', 'NEWOtherComms', 'NEWOvertime', 'NEWPurpleLink',
               'OpSys', 'OrgSize', 'PurchaseWhat', 'Sexuality', 'SOAccount', 'SOComm', 'SOPartFreq', 'SOVisitFreq',
               'SurveyEase', 'SurveyLength', 'Trans', 'UndergradMajor', 'WelcomeChange', 'YearsCode', 'YearsCodePro']
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
    if remained[i] == 'LanguageWorkedWith':
        LanguageWorkedWith = multi_value.keys()
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
no_outlier_columns = ['Respondent', 'CompFreq', 'Country', 'CurrencyDesc', 'CurrencySymbol', 'JobFactors', 'JobSat',
                      'JobSeek', 'LanguageDesireNextYear', 'LanguageWorkedWith', 'MiscTechWorkedWith',
                      'NEWCollabToolsWorkedWith', 'NEWDevOps', 'NEWEdImpt', 'NEWJobHuntResearch', 'NEWLearn',
                      'NEWOffTopic', 'NEWOnboardGood', 'NEWOtherComms', 'NEWOvertime', 'NEWPurpleLink', 'NEWStuck',
                      'OrgSize', 'PlatformDesireNextYear', 'PlatformWorkedWith', 'SOComm', 'SOVisitFreq', 'SurveyEase',
                      'YearsCode', 'YearsCodePro']
# discrete columns (no outlier)
discrete_columns = ['Hobbyist', 'Gender', 'PurchaseWhat', 'Sexuality', 'SOAccount', 'SurveyLength', 'Trans',
                    'UndergradMajor', 'WelcomeChange']

# features with outlier
outlier_columns = list()
for i in encod_ans.columns:
    if i not in (no_outlier_columns + discrete_columns):
        outlier_columns.append(i)

# detecting outliers
"""for i in outlier_columns:
    plt.plot(encod_ans[i], 'o')
    plt.title(i)
    plt.show()"""

# check duplication in Respondent column
boolean = encod_ans.duplicated(subset=['Respondent']).any()

# outlier removal
removable = list()
#encod_ans_no_outlier = pd.DataFrame(columns=encod_ans.columns)
for row in range(len(encod_ans)):
    if encod_ans.loc[row, 'Age'] < 250.0:
        #encod_ans_no_outlier = encod_ans_no_outlier.append(encod_ans.loc[row])
        #removable.append(encod_ans.loc[row, 'Respondent'])
        #encod_ans.drop(encod_ans.loc[row])
        encod_ans.set_index('Respondent', inplace=True, append=True, drop=False).drop(encod_ans.loc[row])
#encod_ans = encod_ans.set_index('Respondent').drop(removable)
outlier_columns.remove('Age')

"""encod_ans_no_outlier = pd.DataFrame(columns=encod_ans.columns)
for i in range(len(encod_ans)):
    for j in removable:
        if encod_ans.loc[i, 'Respondent'] != j:
            encod_ans_no_outlier = encod_ans_no_outlier.append(encod_ans.loc[i])"""

# imputing outlier with median
outliers = []
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
            outliers.append(encod_ans.loc[row])
    median = np.median(encod_ans[column])
    for out in outliers:
        print(out)
        print(column)
        print('------------------------------------------')
        encod_ans.loc[out, column] = median
        #encod_ans_no_outlier = np.where(encod_ans_no_outlier[column] == out, 14, encod_ans_no_outlier[column])

for i in encod_ans.columns:
    plt.plot(encod_ans[i], 'o')
    plt.title(i)
    plt.show()

#sns.scatterplot(x="Age1stCode", y="YearsCode", data=encod_ans)
sns.pairplot(encod_ans, diag_kind="kde")
plt.show()

"""file = open('C:/Users/NIK/Desktop/t.txt', 'r').read().splitlines()
content = list()
for line in file:
    split_line = line.split(';')
    for s in split_line:
        if s not in content:
            content.append(s)
print(len(content), content)"""
