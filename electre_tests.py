import numpy as np
import pandas as pd
from electre import *

def test_marginal_concordance():
    test_alternatives = pd.DataFrame([[90,86,46,30],
    [40,90,14,48],
    [94,100,40,36],
    [78,76,30,50],
    [60,60,30,30],
    [64,72,12,46],
    [62,88,22,48],
    [70,30,12,12]]).astype(float)
    test_boundary_profiles = pd.DataFrame( [[64,61,32,32],
[86,84,43,43]]).astype(float)
    test_indifference_thresholds = [[2,2,0,0],[3,2,0,0]]
    test_preference_thresholds = [[6,5,2,2],[7,8,2,2]]

    test_gains = [True,True,True,True]

    marginal_concordance_alt_to_profile, marginal_concordance_profile_to_alt = calculate_marginal_concordance(test_alternatives, test_boundary_profiles, test_indifference_thresholds, test_preference_thresholds, test_gains)

    concordance_matrix_alt_to_profile = np.array(
        [
        [1,1,1,0],
        [0,1,0,1],
        [1,1,1,1],
        [1,1,0,1],
        [0.5,1,0,0],
        [1,1,0,1],
        [1,1,0,1],
        [1,0,0,0]
        ],
    )
    concordance_matrix_profile_to_alt = np.array(
        [
        [0,0,0,1],
        [1,0,1,0],
        [0,0,0,0],
        [0,0,1,0],
        [1,1,1,1],
        [1,0,1,0],
        [1,0,1,0],
        [0,1,1,1]
        ]
    )
    assert np.allclose(marginal_concordance_alt_to_profile[0], concordance_matrix_alt_to_profile, atol=0.01)
    assert np.allclose(marginal_concordance_profile_to_alt[0], concordance_matrix_profile_to_alt, atol=0.01)

    return marginal_concordance_alt_to_profile, marginal_concordance_profile_to_alt  

def test_marginal_discordance():

    test_alternatives = pd.DataFrame([[90,86,46,30],
    [40,90,14,48],
    [94,100,40,36],
    [78,76,30,50],
    [60,60,30,30],
    [64,72,12,46],
    [62,88,22,48],
    [70,30,12,12]]).astype(float)
    test_boundary_profiles = pd.DataFrame( [[64,61,32,32],
[86,84,43,43]]).astype(float)
    test_veto_thresholds = [[20,24,np.inf,np.inf],[20,25,np.inf,np.inf]]
    test_preference_thresholds = [[6,5,2,2],[7,8,2,2]]

    test_gains = [True,True,True,True]

    discordance_alt_to_profile = np.array([
        [0,0,0,0],
        [1,0,0,0],
        [0,0,0,0],
        [0.07,0,0,0],
        [1,0.94,0,0],
        [1,0.23,0,0],
        [1,0,0,0],
        [0.69,1,0,0]
    ])
    discordance_profile_to_alt = np.array([
        [1,1,0,0],
        [0,1,0,0],
        [1,1,0,0],
        [0.57,0.52,0,0],
        [0,0,0,0],
        [0,0.31,0,0],
        [0,1,0,0],
        [0,0,0,0]
    ])

    marginal_discordance_alt_to_profile, marginal_discordance_profile_to_alt = calculate_marginal_discordance(test_alternatives, test_boundary_profiles, test_preference_thresholds, test_veto_thresholds,test_gains)

    assert np.allclose(marginal_discordance_alt_to_profile[1], discordance_alt_to_profile, atol=0.01)
    assert np.allclose(marginal_discordance_profile_to_alt[0], discordance_profile_to_alt, atol=0.01)

    return marginal_discordance_alt_to_profile, marginal_discordance_profile_to_alt

def test_comprehensive_concordance():

    concordance_matrix_alt_to_profile, concordance_matrix_profile_to_alt = test_marginal_concordance()

    weights = [0.4,0.3,0.25,0.05]

    comprehensive_concordance_alt_to_profile, comprehensive_concordance_profile_to_alt = calculate_comprehensive_concordance(concordance_matrix_alt_to_profile, concordance_matrix_profile_to_alt,weights)
    return comprehensive_concordance_alt_to_profile, comprehensive_concordance_profile_to_alt

def test_outranking_credibility():

    comprehensive_concordance = test_comprehensive_concordance()
    marginal_discordance = test_marginal_discordance()

    outrankings = calculate_outranking_credibility(comprehensive_concordance,marginal_discordance)
    test_outrankings = np.array([[0.95,0],[0.95,0.65]])
    assert np.allclose(outrankings[0], test_outrankings, atol=0.01)
    return outrankings

def test_transform_outranking_to_preference():
    outrankings = test_outranking_credibility()
    outranking_preference = transform_outranking_to_preference(outrankings)
    assert np.allclose(outranking_preference[:,0:1].flatten(), [[1,np.nan,1,1,-1,0,1,np.nan]], equal_nan=True)
    return outranking_preference

def test_class_assignment():

    preference = test_transform_outranking_to_preference()
    classes_pess = class_assignment(preference,3,True)
    classes_opt = class_assignment(preference,3,False)
    assert np.allclose(classes_pess, [3,1,3,2,1,2,2,1])
    assert np.allclose(classes_opt, [3,2,3,2,1,2,2,2])
    return classes_pess, classes_opt

def test_electre_tri_b():
    
    test_alternatives = pd.DataFrame([[70,98,78,76],
    [44,51,23,46],
    [94,100,43,36],
    [78,76,30,50],
    [60,60,30,30],
    [64,72,12,46],
    [62,88,22,48],
    [70,30,12,12]]).astype(float)
    test_boundary_profiles = pd.DataFrame( [[64,61,32,32],
[86,84,43,43]]).astype(float)
    test_veto_thresholds = [[20,24,np.inf,np.inf],[20,25,np.inf,np.inf]]
    test_preference_thresholds = [[6,5,2,2],[7,8,2,2]]
    test_indifference_thresholds = [[2,2,0,0],[3,2,0,0]]
    test_gains = [True,True,True,True]
    test_weights = [0.4,0.3,0.25,0.05]

    classes = electre_tri_b(test_alternatives,test_boundary_profiles,test_gains,test_indifference_thresholds,test_preference_thresholds,test_veto_thresholds,test_weights)
    assert np.allclose(classes, [3,1,3,2,1,2,2,1])
    return classes

