import os 
from project1.promethee.preference_functions import CriterionType, VShapeWithIndifference, MarginalPreferenceMatrix, ComprehensivePreferenceIndex, PrometheeI, PrometheeII  
from data.parse import data, criteria, decision_classes, pairwise_comparisons

PROMETHEE_SAVE_DIR = 'results/promethee' # Directory to save the marginal preference matrices, just for debugging and analysis
os.makedirs(PROMETHEE_SAVE_DIR, exist_ok=True)

CRITERIA_WEIGHTS = {
    'Acting': 5,
    'Plot': 2,
    'Pictures': 3,
    'Music': 7,
    'Sentiment': 5,
    'Critics Score': 3,
    'Oscars Won': 2
}

# TODO: Select more preference functions for each criterion so that they make more sense
pfunction1 = VShapeWithIndifference(0.5, 1.0, CriterionType.GAIN)


# Marginal preference matrix for each criterion
marg_matrices = []
weights = []
for criterion in criteria['Criterion']:
    matrix = MarginalPreferenceMatrix(data[criterion], pfunction1, names=data['Movie'])
    matrix.save(f'{PROMETHEE_SAVE_DIR}/{criterion}_matrix.csv')
    marg_matrices.append(matrix)
    weights.append(CRITERIA_WEIGHTS[criterion])

# Comprehensive preference index
cpi = ComprehensivePreferenceIndex(marg_matrices, weights)
print(cpi)
cpi.save(f'{PROMETHEE_SAVE_DIR}/cpi.csv')

# Create ranking
promethee1 = PrometheeI(cpi)
print(promethee1)
promethee1.plot_ranking(type='positive', savedir=f'{PROMETHEE_SAVE_DIR}/ranking_prom1_positive.png', show=False)
promethee1.plot_ranking(type='negative', savedir=f'{PROMETHEE_SAVE_DIR}/ranking_prom1_negative.png', show=False)
promethee1.plot_ranking(type='overall', savedir=f'{PROMETHEE_SAVE_DIR}/ranking_prom1_overall.png', show=True)

# Do the same for Promethee II
promethee2 = PrometheeII(cpi)
print(promethee2)
promethee2.plot_ranking(type='positive', savedir=f'{PROMETHEE_SAVE_DIR}/ranking_prom2_positive.png', show=False)
promethee2.plot_ranking(type='negative', savedir=f'{PROMETHEE_SAVE_DIR}/ranking_prom2_negative.png', show=False)
promethee2.plot_ranking(type='overall', savedir=f'{PROMETHEE_SAVE_DIR}/ranking_prom2_overall.png', show=True)