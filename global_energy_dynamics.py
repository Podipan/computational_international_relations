'''
Code prepared for the paper: Podiotis, P. (18/10/2020). “Global Energy Dynamics, a Data Analysis Project”. 
International Affairs Forum. https://www.ia-forum.org/Files/HSOCNS.pdf
It includes: Very basic pandas operations, Correlations, Clusterings (Kmeans & Meanshift), Elbow and Silhouette methods, basic visualizations.
Data used were drawn from Podiotis, P. (30/07/2020). "Towards International Relations Data Science: Mining the CIA World Factbook”. Final Paper. 
UoM library repository. https://dspace.lib.uom.gr/handle/2159/24352 and can be also found in the cia_world_factbook repository.
'''

# ----------- IMPORTS -----------
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import MeanShift, KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

#----------- PARAMETERS FOR PANDAS & MATPLOTLIB, can be omitted ---------------------------
plt.rcParams['figure.dpi'] = 100
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)

#------------- THE FUNCTION BELOW LOADS THE DATASET AND DROPS COLUMNS WHICH WILL NOT BE USED, it outputs 2 files, one with countries as rows and the other with continents ----------------------
def Create_Energy_Dataset():
    data = pd.read_pickle(r"DIRECTORY of v5 FACTBOOK FILE")
    energy = pd.DataFrame()
    for c in data.columns:
        if "energy" in c or "pipelines" in c or "oil" in c or "lng" in c:
            energy[c] = data[c]
            if "rural" in c or "urban" in c or "dioxide" in c:
                energy.drop(c, axis="columns", inplace=True)
    energy["continent"] = pd.read_pickle(r"v1 factbook dataset")[
        "lbl Region"]
    energy.drop("sum transportation-pipelines none", axis="columns", inplace=True)
    energy.to_pickle("energy.pkl")
    energy.to_excel("energy.xlsx")

#------------ THE PART BELOW SUMS THE DATA FOR ALL COUNTRIES BELONGING TO THE SAME CONTINENT IN ORDER TO PROVIDE CONTINENT-CUMULATIVE DATA ------------------
    energy = pd.read_pickle("energy.pkl")
    energy.dropna(axis="rows", how="any", inplace=True)
    energy_region = pd.DataFrame()
    for region in energy["continent"].unique():
        temp = energy[energy["continent"] == region]
        temp.drop("continent", axis="columns", inplace=True)
        temp = temp.sum()
        energy_region[region] = temp
    energy_region = energy_region.T
    energy_region.to_excel("energy_regions.xlsx")
    energy_region.to_pickle("energy_regions.pkl")

# ---------------- THE FUNCTION BELOW PRINTS ELBOW AND SILHOUETTE METHOD GRAPHS for estimation of optimal k if Kmeans clustering is used (found online) ------------------
def KMeans_Cluster_Metrics(X):
        # ----------------ELBOW METHOD-----------------
        Sum_of_squared_distances = []
        K = range(1, 15)
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(X)
            Sum_of_squared_distances.append(km.inertia_)
        plt.figure(figsize=(25, 25))
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

        # ---------------------------SILHOUETTE SCORE---------------------
        sil = []
        ks = []
        kmax = 10
        # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
        for k in range(2, kmax + 1):
            kmeans = KMeans(n_clusters=k).fit(X)
            labels = kmeans.labels_
            sil.append(silhouette_score(X, labels, metric='euclidean'))
            ks.append(k)
        plt.plot(np.array(ks), np.array(sil))
        plt.xlabel('Clusters')
        plt.ylabel('Score')
        plt.title('Silhouette Score')
        plt.show()
        
# ------------ CALCULATES THE BIVARIATE CORRELATION BETWEEN THE "ene_columns" AND ALL OTHER COLUMNS OF THE DATASET        
def Create_Correlations():
    data = pd.read_pickle(r"v5 Factbook dataset")
    ener_cols = ["num energy-crude-oil-production", "sum transportation-pipelines oil"]
    cors = pd.DataFrame(index=ener_cols)

    for c in ener_cols:
        for col in data.columns:
            if col not in ener_cols:
                temp = data[[c,col]].copy()
                temp.dropna(axis="rows", how="any", inplace=True)
                y = temp[c].values
                x = temp[col].values
                pearson = pearsonr(x, y)
                spearman = spearmanr(x, y)
                cell = ""
                if pearson[1] < 0.05 and spearman[1] < 0.05 and(abs(pearson[0]) > 0.5 or abs(spearman[0]) > 0.5):
                    cell = str(round(pearson[0], 4))  +  " | " + str(round(spearman[0], 4))
                else:
                    cell = np.nan
                cors.loc[c, col] = cell
    cors.dropna(axis="rows", how="all", inplace=True)
    cors.dropna(axis="columns", how="all", inplace=True)
    print(cors)
    cors.to_excel("Correlations.xlsx")


# ------------- THE MAIN SCRIPT OF THE PAPER ----------------------
# -------------- DATA LOADING AND PREPARATION ---------------------
energy = pd.read_pickle("energy.pkl")
energy.dropna(axis="rows", how="any", inplace=True)
reserves = ["num energy-natural-gas-proved-reserves", "num energy-crude-oil-proved-reserves"]
production = ["num energy-crude-oil-production", "num energy-refined-petroleum-products-production", "num energy-natural-gas-production"]
imports = ["num energy-crude-oil-imports", "num energy-refined-petroleum-products-imports", "num energy-natural-gas-imports"]
exports = ["num energy-crude-oil-exports", "num energy-refined-petroleum-products-exports", "num energy-natural-gas-exports"]
electricity = ["num energy-electricity-from-fossil-fuels", "num energy-electricity-from-nuclear-fuels", "num energy-electricity-from-hydroelectric-plants",	"num energy-electricity-from-other-renewable-sources"]
terminals = ["amount transportation-ports-and-terminals lng terminal(s) (import)","amount transportation-ports-and-terminals lng terminal(s) (export)","amount transportation-ports-and-terminals oil terminal(s)"]
pipelines = ["sum transportation-pipelines condensate",	"sum transportation-pipelines gas","sum transportation-pipelines liquid petroleum gas","sum transportation-pipelines oil",
             "sum transportation-pipelines refined products","sum transportation-pipelines chemicals"]
all_columns = reserves + production + imports + exports + electricity

all = pd.read_pickle("energy.pkl")
all["name"] = pd.read_pickle(r"v1 factbook dataset")["txt Country Name"]
all["continent"] = all["continent"].map({'central america':"Central America", 'middle east':"Middle East", 'south asia':"S. Asia", "africa":"Africa",
                     'europe':"Europe", 'australia - oceania':"Oceania", 'south america':"S. America",'north america':"N. America",
                     'east asia/southeast asia':"E./S.E. Asia", 'central asia':"Central Asia"})
all.drop("continent", axis="columns", inplace=True)
energy = pd.DataFrame(all)

# ---------------- MEASURE UNITS CONVERSIONS for DATASET WITH COUNTRIES -------------------------
energy["num energy-natural-gas-production"] = energy["num energy-natural-gas-production"]/26.137 #GJoules
energy["num energy-refined-petroleum-products-production"] = energy["num energy-refined-petroleum-products-production"]*6.33*365 #GJoules
energy["num energy-crude-oil-production"] = energy["num energy-crude-oil-production"]*6.33*365 #GJoules

energy["num energy-natural-gas-proved-reserves"] = energy["num energy-natural-gas-proved-reserves"]/26.137 #GJoules
energy["num energy-crude-oil-proved-reserves"] = energy["num energy-crude-oil-proved-reserves"]*6.33 #GJoules

energy["num energy-natural-gas-imports"] = energy["num energy-natural-gas-imports"]/26.137 #GJoules
energy["num energy-refined-petroleum-products-imports"] = energy["num energy-refined-petroleum-products-imports"]*6.33*365 #GJoules
energy["num energy-crude-oil-imports"] = energy["num energy-crude-oil-imports"]*6.33*365 #GJoules

energy["num energy-natural-gas-exports"] = energy["num energy-natural-gas-exports"]/26.137 #GJoules
energy["num energy-refined-petroleum-products-exports"] = energy["num energy-refined-petroleum-products-exports"]*6.33*365 #GJoules
energy["num energy-crude-oil-exports"] = energy["num energy-crude-oil-exports"]*6.33*365 #GJoules
energy = energy[all_columns]
print(energy.shape)

# ---------------- MEASURE UNITS CONVERSIONS for DATASET WITH CONTINENTS -------------------------
energy = pd.read_pickle("energy_regions.pkl")
energy.rename(index={'central america':"Central America", 'middle east':"Middle East", 'south asia':"S. Asia", "africa":"Africa",
                     'europe':"Europe", 'australia - oceania':"Oceania", 'south america':"S. America",'north america':"N. America",
                     'east asia/southeast asia':"E./S.E. Asia", 'central asia':"Central Asia"}, inplace=True)

energy = energy[all_columns]
print(energy.columns)

energy["num energy-natural-gas-production"] = energy["num energy-natural-gas-production"]/26.137 #GJoules
energy["num energy-refined-petroleum-products-production"] = energy["num energy-refined-petroleum-products-production"]*6.33*365 #GJoules
energy["num energy-crude-oil-production"] = energy["num energy-crude-oil-production"]*6.33*365 #GJoules

energy["num energy-natural-gas-proved-reserves"] = energy["num energy-natural-gas-proved-reserves"]/26.137 #GJoules
energy["num energy-crude-oil-proved-reserves"] = energy["num energy-crude-oil-proved-reserves"]*6.33 #GJoules

energy["num energy-natural-gas-imports"] = energy["num energy-natural-gas-imports"]/26.137 #GJoules
energy["num energy-refined-petroleum-products-imports"] = energy["num energy-refined-petroleum-products-imports"]*6.33*365 #GJoules
energy["num energy-crude-oil-imports"] = energy["num energy-crude-oil-imports"]*6.33*365 #GJoules

energy["num energy-natural-gas-exports"] = energy["num energy-natural-gas-exports"]/26.137 #GJoules
energy["num energy-refined-petroleum-products-exports"] = energy["num energy-refined-petroleum-products-exports"]*6.33*365 #GJoules
energy["num energy-crude-oil-exports"] = energy["num energy-crude-oil-exports"]*6.33*365 #GJoules


#------------------COUNTRY AVERAGE VALUES CALCULATION------------------
print(all["continent"])
for i in energy.index:
    for c in energy.columns:
        energy.loc[i,c] = energy.loc[i,c] /len(all[all["continent"] == i])

#------------GRAPH CREATION-----------
print(energy.columns)
energy.plot(kind='bar', alpha=0.75, rot=25)
plt.title("Country-Average Pipelines")
plt.ylabel("Kilometers")
#plt.figtext(0.1, 0.01, "", fontsize=12)
plt.show()


#--------------------------- SCALING & CLUSTERING -------------------------
energy.dropna(axis="rows", how="any", inplace=True)
print(energy.shape)

scaler = MinMaxScaler()
energy = pd.DataFrame(scaler.fit_transform(energy), columns=energy.columns, index=energy.index)
X = energy.values
energy["label"] = MeanShift().fit_predict(X) # change here for Kmeans
energy["name"] = pd.read_pickle(r"v1 factbook dataset")["txt Country Name"]
print("CLUSTERS FOUND: {}".format(max(energy["label"].values)))

# ------------------ DESCRIPTIVE STATISTICS FOR EACH CLUSTER ----------------------
for i in energy["label"].unique():
    temp = energy[energy["label"] == i]
    temp.drop("label", axis="columns", inplace=True)
    print(temp["name"].values.tolist(), "\n",temp.describe().sort_values(by=["mean", "count"], axis=1, ascending=False))

