import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def concordance_function(difference,indifference,preference,gain):
    if gain:
        if difference >= -indifference:
            return 1
        elif difference < -preference:
            return 0
        else:
            return (preference + difference) / (preference-indifference)
    else:
        if difference <= indifference:
            return 1
        elif difference > preference:
            return 0
        else:
            return (preference-difference) / (preference-indifference)


def calculate_marginal_concordance(alternatives, boundary_profiles, indifference_thresholds, preference_thresholds, gain,concordance_function=concordance_function):

    marginal_concordance_alt_to_profile = np.zeros((len(boundary_profiles),len(alternatives),len(alternatives.columns),))
    marginal_concordance_profile_to_alt = np.zeros((len(boundary_profiles),len(alternatives),len(alternatives.columns),))

    for i in range(len(alternatives)):
        for j in range(len(boundary_profiles)):
            for k in range(len(alternatives.columns)):
                difference = alternatives.iloc[i,k] - boundary_profiles.iloc[j,k]
                
                indifference = indifference_thresholds[j][k]
                preference = preference_thresholds[j][k]
                marginal_concordance_alt_to_profile[j,i,k] = concordance_function(difference, indifference, preference, gain[k])
                marginal_concordance_profile_to_alt[j,i,k] = concordance_function(-difference, indifference, preference, gain[k])

    return marginal_concordance_alt_to_profile,marginal_concordance_profile_to_alt

def discordance_function(difference, veto, preference, gain):
    if gain:
        if difference <= -veto:
            return 1
        elif difference >= -preference:
            return 0
        else :
            return (-difference - preference) / (veto-preference)
    else:
        if difference >= veto:
            return 1
        elif difference <= preference:
            return 0
        else:
            return (veto - difference) / (veto-preference)

def calculate_marginal_discordance(alternatives,boundary_profiles,preference_thresholds,veto_thresholds,gain,discordance_function=discordance_function):

    marginal_discordance_alt_to_profile = np.zeros((len(boundary_profiles),len(alternatives),len(alternatives.columns),))
    marginal_discordance_profile_to_alt = np.zeros((len(boundary_profiles),len(alternatives),len(alternatives.columns),))

    for i in range(len(alternatives)):
        for j in range(len(boundary_profiles)):
            for k in range(len(alternatives.columns)):
                difference = alternatives.iloc[i,k] - boundary_profiles.iloc[j,k]
                
                veto = veto_thresholds[j][k]
                preference = preference_thresholds[j][k]
                marginal_discordance_alt_to_profile[j,i,k] = discordance_function(difference, veto, preference, gain[k])
                marginal_discordance_profile_to_alt[j,i,k] = discordance_function(-difference, veto, preference, gain[k])

    return marginal_discordance_alt_to_profile,marginal_discordance_profile_to_alt

def calculate_comprehensive_concordance(concordance_matrix_alt_to_profile, concordance_matrix_profile_to_alt,weights):

    comprehensive_concordance_alt_to_profile = np.average(concordance_matrix_alt_to_profile, axis=2, weights=weights)
    comprehensive_concordance_profile_to_alt = np.average(concordance_matrix_profile_to_alt, axis=2, weights=weights)

    return comprehensive_concordance_alt_to_profile.T, comprehensive_concordance_profile_to_alt.T

def calculate_outranking_credibility(comprehensive_concordance,marginal_discordance):
    outrankings = np.zeros((len(comprehensive_concordance[0]),len(comprehensive_concordance[0][0]),2))
    for i in range(len(comprehensive_concordance[0])): # numver of alternatives
        for j in range(len(comprehensive_concordance[0][0])): # number of boundary profiles
            outranking = [comprehensive_concordance[0][i][j], comprehensive_concordance[1][i][j]]
            for k in range(len(marginal_discordance[0])): # number of criteria
                
                if comprehensive_concordance[0][i][j] < marginal_discordance[0][j][i][k]: # alt_to_profile
                    outranking[0] *= (1-marginal_discordance[0][j][i][k])/(1-comprehensive_concordance[0][i][j])
                if comprehensive_concordance[1][i][j] < marginal_discordance[1][j][i][k]: # profile_to_alt
                    outranking[1] *= (1-marginal_discordance[1][j][i][k])/(1-comprehensive_concordance[1][i][j])
            outrankings[i][j] = outranking
    
    return outrankings

def preference_aggregation(aPb,bPa):
    if aPb and not bPa:
        return 1
    if not aPb and bPa:
        return -1
    if aPb and bPa: 
        return 0
    if not aPb and not bPa:
        return None

def transform_outranking_to_preference(outrankings,credibility_threshold=0.65):

    outranking_preference = np.zeros((len(outrankings),len(outrankings[0])))
    for i in range(len(outrankings)):
        for j in range(len(outrankings[0])):
            outranking_preference[i][j] = preference_aggregation(outrankings[i][j][0] >= credibility_threshold, outrankings[i][j][1] >= credibility_threshold)
    return outranking_preference

def class_assignment(preference_matrix,num_classes, pessimistic=True):
    if pessimistic:
        classes = np.full(len(preference_matrix),num_classes)

    else:
        classes = np.ones(len(preference_matrix))


    for j,a in enumerate(preference_matrix):
        if pessimistic:
            for i in range(len(a)-1,-1,-1):
                
                if a[i] >=0:
                    break
                classes[j]-=1
            
        else:
            for i in range(len(a)):
                if a[i] == -1:
                    break
                classes[j]+=1
                
    return classes 
 
def electre_tri_b(alternatives,boundary_profiles,criteria_is_gain,indifference_thresholds,preference_thresholds,veto_thresholds,weights,pessimistic=True):

    concordance_matrix_alt_to_profile, concordance_matrix_profile_to_alt = calculate_marginal_concordance(alternatives,boundary_profiles,indifference_thresholds,preference_thresholds,criteria_is_gain)
    discordance_matrix_alt_to_profile, discordance_matrix_profile_to_alt = calculate_marginal_discordance(alternatives,boundary_profiles,preference_thresholds,veto_thresholds,criteria_is_gain)

    comprehensive_concordance = calculate_comprehensive_concordance(concordance_matrix_alt_to_profile,concordance_matrix_profile_to_alt,weights)

    outrankings = calculate_outranking_credibility(comprehensive_concordance,[discordance_matrix_alt_to_profile,discordance_matrix_profile_to_alt])

    outranking_preference = transform_outranking_to_preference(outrankings)

    classes = class_assignment(outranking_preference,len(boundary_profiles)+1,pessimistic)

    return classes

def plot_electre_classes(data, class_column):
    G = nx.DiGraph()
    for name, level in zip(data['Movie'], data[class_column]):
        G.add_node(name, level=level)
        print(int(level), end=' ')
    print()

    pos = nx.multipartite_layout(G, subset_key="level", align='horizontal')
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=7, font_weight='bold')    
    plt.show()

 