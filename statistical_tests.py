import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import pingouin as pt
from pingouin import ttest
import numpy as np

# Load the high five data and handshake data
high_five_data = pd.read_csv('results_mse/high_five_after_training_info.csv')
handshake_data = pd.read_csv('results_mse/handshake_after_training_info.csv')
high_five_data['Greeting_type'] = 'high_five'
handshake_data['Greeting_type'] = 'handshake'


high_five_ascending = high_five_data.loc[high_five_data['Order_type'] == 'ascending']
high_five_descending = high_five_data.loc[high_five_data['Order_type'] == 'descending']
high_five_random = high_five_data.loc[high_five_data['Order_type'] == 'random']

handshake_ascending = handshake_data.loc[handshake_data['Order_type'] == 'ascending']
handshake_descending = handshake_data.loc[handshake_data['Order_type'] == 'descending']
handshake_random = handshake_data.loc[handshake_data['Order_type'] == 'random']


def t_test_orders(question):
    # t-test for random and ascending
    print(f'high five random ascending', question)
    ids_list = high_five_ascending['Participant_number'].tolist()
    high_five_random_filtered = high_five_random.loc[high_five_random['Participant_number'].isin(ids_list)]

    if question == 'Mean_MSE':
        res = ttest(x=high_five_ascending[question], y=high_five_random_filtered[question], paired=True)
    else:
        res = ttest(x=high_five_ascending['Teaching_score'], y=high_five_random['Teaching_score'], paired=True,
                    correction=True)
    pt.print_table(res, floatfmt='.5f')

    print('*******************')


    # t_test for random and descending
    print(f'high five random descending', question)
    ids_list = high_five_descending['Participant_number'].tolist()
    high_five_random_filtered = high_five_random.loc[high_five_random['Participant_number'].isin(ids_list)]
    res = ttest(x=high_five_descending[question], y=high_five_random_filtered[question], paired=True)
    pt.print_table(res, floatfmt='.5f')

    print('*******************')

    # t-test for ascending and descending
    print(f'high five ascending descending', question)
    ids_list_asc = high_five_ascending['Participant_number'].tolist()
    ids_list_des = high_five_descending['Participant_number'].tolist()
    if len(ids_list_asc) >= len(ids_list_des):
        ids_amount = len(ids_list_des)
        high_five_descending_filtered = high_five_descending
        ids_list = np.random.choice(ids_list_asc, ids_amount, replace=False)
        high_five_ascending_filtered = high_five_ascending.loc[high_five_ascending['Participant_number'].isin(ids_list)]
    else:
        ids_amount = len(ids_list_asc)
        high_five_ascending_filtered = high_five_ascending
        ids_list = np.random.choice(ids_list_des, ids_amount, replace=False)
        high_five_descending_filtered = high_five_descending.loc[high_five_descending['Participant_number'].isin(ids_list)]

    res = ttest(x=high_five_ascending_filtered[question], y=high_five_descending_filtered[question], paired=False)
    pt.print_table(res, floatfmt='.5f')



    # t-test for random and ascending
    print(f'handshake random ascending ', question)
    ids_list = handshake_ascending['Participant_number'].tolist()
    handshake_random_filtered = handshake_random.loc[handshake_random['Participant_number'].isin(ids_list)]
    res = ttest(x=handshake_ascending[question], y=handshake_random_filtered[question], paired=True)
    pt.print_table(res, floatfmt='.5f')

    print('*******************')


    # t_test for random and descending
    print(f'handshake random descending', question)
    ids_list = handshake_descending['Participant_number'].tolist()
    handshake_random_filtered = handshake_random.loc[handshake_random['Participant_number'].isin(ids_list)]
    res = ttest(x=handshake_descending[question], y=handshake_random_filtered[question], paired=True)
    pt.print_table(res, floatfmt='.5f')

    print('*******************')

    print(f'handshake ascending descending ', question)
    ids_list_asc = handshake_ascending['Participant_number'].tolist()
    ids_list_des = handshake_descending['Participant_number'].tolist()
    if len(ids_list_asc) >= len(ids_list_des):
        ids_amount = len(ids_list_des)
        handshake_descending_filtered = handshake_descending
        ids_list = np.random.choice(ids_list_asc, ids_amount, replace=False)
        handshake_ascending_filtered = handshake_ascending.loc[handshake_ascending['Participant_number'].isin(ids_list)]
    else:
        ids_amount = len(ids_list_asc)
        handshake_ascending_filtered = handshake_ascending
        ids_list = np.random.choice(ids_list_des, ids_amount, replace=False)
        handshake_descending_filtered = handshake_descending.loc[handshake_descending['Participant_number'].isin(ids_list)]

    res = ttest(x=handshake_ascending_filtered[question], y=handshake_descending_filtered[question], paired=False)
    pt.print_table(res, floatfmt='.5f')


questions = ['Mean_MSE', 'Teaching_score']

# For High Five data
for question in questions:
    t_test_orders(question)


# Load the high five data and handshake data
high_five_data = pd.read_csv('results_mse/high_five_after_training_info.csv')
handshake_data = pd.read_csv('results_mse/handshake_after_training_info.csv')
high_five_data['Greeting_type'] = 'high_five'
handshake_data['Greeting_type'] = 'handshake'

# Combine the DataFrames
combined_data = pd.concat([high_five_data, handshake_data], ignore_index=True)


# Read MSE data from CSV files for Handshake and High Five separately
handshake_mse_data = pd.read_csv('results_mse/handshake_mse.csv', header=None)
high_five_mse_data = pd.read_csv('results_mse/high_five_mse.csv', header=None)

# Extract MSE values from the datasets
mse_list_handshake = handshake_mse_data[0].values
mse_list_high_five = high_five_mse_data[0].values

# Perform a t-test for independent samples (assuming both lists have equal variance)
t_stat, p_value_ttest = ttest_ind(mse_list_handshake, mse_list_high_five)


# Perform a Mann-Whitney U test (non-parametric test) for independent samples
u_stat, p_value_mwu = mannwhitneyu(mse_list_handshake, mse_list_high_five, alternative='two-sided')

# Print the results
print("T-test results:")
print(f"T-statistic: {t_stat}")
print(f"P-value: {p_value_ttest}")

print("\nMann-Whitney U test results:")
print(f"U-statistic: {u_stat}")
print(f"P-value: {p_value_mwu}")


total_teaching_score_per_participant = combined_data.groupby('Participant_number')['Teaching_score'].sum()
total_teaching_score_per_greeting = combined_data.groupby(['Participant_number', 'Greeting_type'])['Teaching_score'].sum()

# Calculate average total teaching score and average teaching score per greeting
average_total_teaching_score = total_teaching_score_per_participant.mean()
average_teaching_score_per_greeting = total_teaching_score_per_greeting.groupby('Greeting_type').mean()

# Print the results
print("Total Teaching Score per Participant:")
print(total_teaching_score_per_participant)

print("\nTotal Teaching Score per Participant for each Greeting:")
print(total_teaching_score_per_greeting)

print("\nAverage Total Teaching Score:")
print(average_total_teaching_score)

print("\nAverage Teaching Score per Greeting:")
print(average_teaching_score_per_greeting)


