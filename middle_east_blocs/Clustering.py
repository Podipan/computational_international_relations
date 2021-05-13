import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import regex as re
import scipy
from scipy.stats import gmean, mannwhitneyu, levene
from sklearn.cluster import KMeans, MeanShift
pd.set_option('display.width', 500)
pd.set_option('display.max_columns',10)
'''
MNAR = ["amount government-international-organization-participation", "amount geography-land-boundaries-border-countries-overall"]
cols = ["amount terrorism-terrorist-groups-home-based", "amount terrorism-terrorist-groups-foreign-based",
        "amount people-and-society-ethnic-groups", "amount government-international-organization-participation"
        "amount transportation-ports-and-terminals major seaport(s)", "sum transportation-pipelines gas", "sum transportation-pipelines oil", "sum transportation-pipelines refined products",
        "sum transnational-issues-refugees-and-internally-displaced-persons refugees (country of origin)",
        "amount geography-land-boundaries-border-countries-overall", "sum transportation-merchant-marine by type oil tanker",
        "amount military-and-security-military-branches", "num geography-coastline", "num geography-land-use agricultural land",
        "num people-and-society-population", "num people-and-society-urbanization urban population", "num people-and-society-infant-mortality-rate total",
        "num people-and-society-drinking-water-source total", "num economy-gdp-purchasing-power-parity hist", "num economy-gdp-per-capita-ppp hist",
        "num economy-gdp-composition-by-end-use government consumption", "num economy-labor-force"]
        '''
data = pd.read_pickle(r"Master Imputed.pkl")

blocs = data.loc[["tu", "qa", "eg", "ae", "sa"]]
blocs.dropna(axis="columns", how="any", inplace=True)
nunique = blocs.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
blocs.drop(cols_to_drop, axis=1, inplace=True)
print(blocs.shape)
#---------------original------------------
bloc1_orig = data.loc[["tu", "qa"]]
bloc2_orig = data.loc[["eg", "ae", "sa"]]


#---------------scaled----------------
scaler = MinMaxScaler()
temp = pd.DataFrame(scaler.fit_transform(blocs), columns=blocs.columns, index=blocs.index)

all = temp
bloc1_scaled = temp.loc[["tu", "qa"]]
bloc2_scaled = temp.loc[["eg", "ae", "sa"]]
print(len(all.columns))
bloc1_hom = bloc1_scaled.describe()
print(sum(bloc1_hom.loc["std"].values))

bloc2_hom = bloc2_scaled.describe()
print(sum(bloc2_hom.loc["std"].values))

bloc1_hom.to_excel("bloc1.xlsx")
bloc2_hom.to_excel("bloc2.xlsx")
#-------------form blocks blocs--------------
bloc1 = pd.DataFrame(bloc1_scaled.mean(axis=0)).T
bloc1.rename(index={0:"bloc1"}, inplace=True)
bloc2 = pd.DataFrame(index=["bloc2"], columns=bloc2_scaled.columns)
for c in bloc2_scaled.columns:
    if 0 in bloc2_scaled[c].values:
        mean_val = sum(bloc2_scaled[c].values)/len(bloc2_scaled[c].values)
        bloc2[c] = mean_val
    else:
        mean_val = gmean(bloc2_scaled[c].values)
        bloc2[c] = mean_val

df = pd.DataFrame(columns=bloc1.columns)
df = df.append([bloc1, bloc2])

def factbook_subsets_Creator():
    data = pd.read_pickle(r"Master Imputed.pkl")
    names_regions = pd.read_pickle(r"MASTER.pkl")[["lbl Region", "txt Country Name"]]
    data[["lbl Region", "txt Country Name"]] = names_regions

    regions_cum = pd.DataFrame()
    regions_avg = pd.DataFrame()

    for region in data["lbl Region"].unique():
        temp = data[data["lbl Region"] == region]
        temp.drop(["lbl Region", "txt Country Name"], axis="columns", inplace=True)

        regions_cum[region] = temp.sum()
        regions_avg[region] = temp.sum()/temp.shape[0]

        if region == "middle east":
            temp["txt Country Name"] = names_regions["txt Country Name"]
            temp.to_excel("middle east.xlsx")
            temp.to_pickle("middle east.pkl")

    regions_cum.T.to_excel("regions_cum.xlsx")
    regions_avg.T.to_excel("regions_avg.xlsx")
    regions_cum.T.to_pickle("regions_cum.pkl")
    regions_avg.T.to_pickle("regions_avg.pkl")
def get_best_distribution(data):
    import scipy.stats as st
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        #print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    #print("Best fitting distribution: "+str(best_dist))
    #print("Best p value: "+ str(best_p))
    #print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, #params[best_dist]

for i in df.index:
    dist, p = get_best_distribution(df.loc[i].values)
    print(dist, p)

x = df.loc["bloc1"].values
y =  df.loc["bloc2"].values
#print(mannwhitneyu(x,y))
#print(levene(x,y))

method = MeanShift()
predictions = method.fit_predict(all.values)
all["cluster"] = predictions
print(all["cluster"])
colors=["r", "g", "b"]
for l, col in zip(all["cluster"].unique(), colors):
    temp = all[all["cluster"] == l]
    desc = temp.describe().sort_values(by=["mean", "count"], axis=1, ascending=False)
    name = "cluster" + str(l) + ".xlsx"
    desc.to_excel(name)

