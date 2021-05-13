import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import LabelEncoder

def Factbook_religions_creation():
    data = pd.read_pickle("MASTER.pkl")
    df = pd.DataFrame(index=data.index)
    df.drop("kn", axis="rows", inplace=True)
    df.drop("wi", axis="rows", inplace=True)

    for i in df.index:
        if not pd.isnull(data.loc[i, "txt government-dependency-status"]):
            df.drop(i, axis="rows", inplace=True)

    print(df.shape)
    c = "txt people-and-society-religions"
    for i in df.index:
        cell = data.loc[i,c]
        if not pd.isnull(cell):
            if "note" in cell:
                cell = cell.split("note")[0]
            cell = re.sub('\([^)]*\)', "", cell)
            cell = cell.split(",")
            df.loc[i, "amount religions"] = len(cell)
            #cell = re.findall("\D+ \d+\.*\d+%", cell)
            for item in cell:
                religion = re.sub("[\d%.\-]|&lt;|to|nominally|approximately|more than", "", item).strip()
                value = re.sub("[a-z%/)&;']", "", item).strip()
                if "-" in value:
                    value = value.split("-")[-1].strip()
                if "  " in value:
                    value = value.split("  ")[-1].strip()

                try:
                    value = round(float(value), 2)
                except:
                    value = np.nan

                df.loc[i, "Religion - " + religion] = value

        else:
            df.drop(i, axis="rows", inplace=True)

    df.fillna(0, inplace=True)
    for i in df.index:
        cell = data.loc[i,c]
        if "note" in cell:
            cell = cell.split("note")[0]
        cell = re.sub('\([^)]*\)', "", cell).strip()

        if i == "sa":
            print(df.loc[i].values)

        temp = df.loc[i].values.tolist()
        del temp[0]

        if sum(temp) == 0:
            if len(cell.split(",")) < 2:
                religion = re.sub('\([^)]*\)|', "", cell).strip()
                religion = re.sub('[;)(]', "", religion).strip()
                df.loc[i, "Religion - " + religion] = 100
            else:
                df.loc["ek", "Religion - " + "roman catholic"] = 88
                df.loc["ek", "Religion - " + "protestant"] = 5
                df.loc["ek", "Religion - " + "sunni muslim"] = 2
                df.loc["ek", "Religion - " + "other"] = 5

                df.loc["er", "Religion - " + "christian"] = 68
                df.loc["er", "Religion - " + "muslim"] = 37
                df.loc["er", "Religion - " + "other"] = 1

                df.loc["mv", "Religion - " + "muslim"] = 100

                df.loc["sm", "Religion - " + "catholic"] = 100

                df.loc["vt", "Religion - " + "catholic"] = 100

                df.loc["so", "Religion - " + "muslim"] = 97
                df.loc["so", "Religion - " + "other"] = 3


                df.loc["gt", "Religion - " + "catholic"] = 45
                df.loc["gt", "Religion - " + "protestant"] = 42
                df.loc["gt", "Religion - " + "other"] = 13

                df.loc["ma", "Religion - " + "christian"] = 41
                df.loc["ma", "Religion - " + "muslim"] = 7
                df.loc["ma", "Religion - " + "indigenous religionist"] = 52

                df.loc["od", "Religion - " + "christian"] = 60
                df.loc["od", "Religion - " + "muslim"] = 6
                df.loc["od", "Religion - " + "indigenous religionist"] = 33

                df.loc["sa", "Religion - " + "sunni muslim"] = 85
                df.loc["sa", "Religion - " + "shia muslim"] = 10
                df.loc["sa", "Religion - " + "other"] = 5

                df.loc["su", "Religion - " + "sunni muslim"] = 88
                df.loc["su", "Religion - " + "shia muslim"] = 1
                df.loc["su", "Religion - " + "indigenous religionist"] = 1.5
                df.loc["su", "Religion - " + "christian"] = 1.5
                df.loc["su", "Religion - " + "other"] = 4

                df.loc["up", "Religion - " + "christian"] = 85
                df.loc["up", "Religion - " + "protestant"] = 2.5
                df.loc["up", "Religion - " + "unaffiliated"] = 12


    df["Religion - atheist/unknown/various"] = 0
    df["Religion - indigenous/folk"] = 0
    df.fillna(0, inplace=True)
    for c in df.columns:
        if "orthodox" in c and c != "Religion - orthodox":
            df["Religion - orthodox"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif "protestant" in c and c != "Religion - protestant":
            df["Religion - protestant"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif ("christian" in c or "church" in c or "kimbanguist" in c) and (c != "Religion - christian"):
            df["Religion - christian"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif "other" in c and c != "Religion - other":
            df["Religion - other"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif ("atheist" in c or "no" in c or "unspecified" in c or "unaffiliated" in c or "objected" in c) and (c != "Religion - atheist/unknown/various"):
            df["Religion - atheist/unknown/various"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif "catholic" in c and c != "Religion - catholic":
            df["Religion - catholic"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif "evangelical" in c and c != "Religion - evangelical":
            df["Religion - evangelical"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif ("folk" in c or "indigenous" in c or "traditional" in c or "tribal" in c or "cusmary" in c) and (c != "Religion - indigenous/folk"):
            df["Religion - indigenous/folk"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif "muslim" in c:
            df["Religion - islam"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif "budd" in c and c != "Religion - buddhism":
            df["Religion - buddhism"] += df[c]
            df.drop(c, axis="columns", inplace=True)
        elif "hindu" in c and c != "Religion - hindu":
            df["Religion - hindu"] += df[c]
            df.drop(c, axis="columns", inplace=True)

    df["Religion - atheist/unknown/various"] += df["Religion - other"]
    df.drop("Religion - other", axis="columns", inplace=True)
    for i in df.index.values:
        suma = sum(df.loc[i].values.tolist())
        if suma<95:
            df.loc[i, "Religion - atheist/unknown/various"] += 100 - suma

    df.replace(0,np.nan, inplace=True)
    print(df.shape)
    df.dropna(axis="columns", how="all", inplace=True)
    df.fillna(0, inplace=True)
    df.to_excel("Factbook_religions.xlsx")
    df.to_pickle("Factbook_religions.pkl")
    print(df.shape)
def Dominant_Religions_Creator():
    data = pd.read_pickle("Factbook_religions.pkl")
    data.drop("dominant - religion", axis="columns", inplace=True)
    religions = data.idxmax(axis=1)
    diction = dict()
    for i in data.index:
        maxi = max(data.loc[i])
        if maxi > 50:
            religion = religions[i].split("-")[-1].strip()
            data.loc[i, "dominant - religion"] = religion
            if religion not in diction.keys():
                diction[religion] = 0
            diction[religion] += 1

    data["dominant - religion"].fillna("no dominant", inplace=True)
    data.to_excel("Factbook_religions.xlsx")
    data.to_pickle("Factbook_religions.pkl")


Factbook_religions_creation()

def Global_Religions_bar():
    plt.style.use('ggplot')
    x = [31.4,23.2,15,7.1,0.2,6.7,16.4]
    y = ["christian", "muslim", "hindu", "buddhist", "jewish", "other", "unaffiliated"]
    colors = ['r','g','b','k','y','m','c',"gray","o"]
    x_pos = [i for i, _ in enumerate(x)]
    for i in range(len(x)):
        plt.bar(x_pos[i], x[i], color=colors[i])
    plt.xticks(x_pos, y)
    #Set descriptions:
    plt.title("World Religions")
    plt.ylabel('Global population %')
    plt.xlabel('Religions')
    plt.show()
def Dominant_Religions_Graph():
    plt.style.use('ggplot')
    x = [50, 41, 18, 16, 11, 9, 7]
    y = ['Islam', 'Catholic', 'Christian', 'Protestant', 'Atheist/Various/Local', 'Orthodox', 'Buddhism']
    colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', "gray", "o"]
    x_pos = [i for i, _ in enumerate(x)]
    for i in range(len(x)):
        plt.bar(x_pos[i], x[i], color=colors[i])
    plt.xticks(x_pos, y)
    # Set descriptions:
    plt.title("Top Country - Dominant Religions")
    plt.ylabel('Amount of Countries')
    plt.show()
def Religion_per_Freedom():
    data = pd.read_pickle("MASTER_reduced_encoded.pkl")
    df = pd.DataFrame(data[["txt Country Name", "lbl government-government-type"]])
    data = pd.read_pickle("Factbook_religions.pkl")
    df = df.join(data["dominant - religion"])
    print(df.shape)
    df.dropna(axis="rows", how="any", inplace=True)
    print(df.shape)

    c = "lbl government-government-type"
    for i in df.index.values:
        cell = df.loc[i, c]
        print("\n", cell)
        if "dictator" in cell or "authoritar" in cell:
            df.loc[i, "num_lbl government-government-type"] = 0
            df.loc[i, c] = "authoritarian"
        elif "absolut" in cell or "theocr" in cell:
            df.loc[i, "num_lbl government-government-type"] = 1
            df.loc[i, c] = "absolute/theocratic"
        elif "communis" in cell:
            df.loc[i, "num_lbl government-government-type"] = 2
            df.loc[i, c] = "communist"
        elif "transit" in cell:
            df.loc[i, "num_lbl government-government-type"] = 3
            df.loc[i, c] = "transition/failed"
        elif "democra" in cell or "republ" in cell:
            df.loc[i, "num_lbl government-government-type"] = 5
            df.loc[i, c] = "democratic/republic"
        elif "monarch" in cell:
            df.loc[i, "num_lbl government-government-type"] = 4
            df.loc[i, c] = "monarchy"
        else:
            df.loc[i, "num_lbl government-government-type"] = 5
            df.loc[i, c] = "democratic/republic"
        print(df.loc[i, c], df.loc[i, "num_lbl government-government-type"])
    print(df.shape)
    df.to_excel("vis.xlsx")

    assistant = dict()
    for i in df["lbl government-government-type"].unique():
        country_d = dict()
        x = df.loc[df["lbl government-government-type"] == i]
        for rel in x["dominant - religion"].unique():
            incr = x.loc[x["dominant - religion"] == rel].shape[0]
            country_d[rel] = int(incr)
        print(i, country_d)

    title = "Dominant Religions of Communist Regimes"
    d = {'various/none': 2, 'catholic': 1, 'buddhism': 1}
    labels = list(d.keys())
    sizes = list(d.values())
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(title)

    plt.show()

def Correlations():
    rel = pd.read_pickle("Factbook_religions.pkl")
    print(rel.shape)
    data = pd.read_pickle("Master Imputed.pkl")
    print(data.shape)
    df = rel.join(data)
    print(df.shape)
    correlation = df.corr("spearman")
    correlation.to_excel("vis.xlsx")
    i = "amount religions"
    for c in correlation.columns:
        if "Religion" not in c and abs(correlation.loc[i, c]) > 0.3 and abs(correlation.loc[i, c]) != 1:
            print(i, " is correlated with: ", c, "at: ", correlation.loc[i, c])

    #scipy.stats.pearsonr(x_, y_)

def Gini_Religions():
    from scipy.stats import kendalltau
    rel = pd.read_pickle("Factbook_religions.pkl")
    print(rel.shape)
    data = pd.read_pickle("Master Imputed.pkl")
    print(data.shape)
    df = rel.join(data)
    rel = pd.read_pickle("MASTER.pkl")
    df = df.join(rel["lbl Region"])

    plt.scatter( df["Religion - islam"], df["amount terrorism-terrorist-groups-home-based"],)
    plt.xlabel('rel')
    plt.show()

    d = {}
    for region in df["lbl Region"].unique():
        x = df[df["lbl Region"] == region]["amount religions"]
        avg = sum(x.values)/len(x)
        d[region] = round(avg, 2)
        print(d)
    plt.style.use('ggplot')
    x = list(d.values())
    y = list(d.keys())
    colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c', "gray", "tan", "darkolivegreen"]
    x_pos = [i for i, _ in enumerate(x)]
    for i in range(len(x)):
        plt.bar(x_pos[i], x[i], color=colors[i], )
    plt.xticks(x_pos, y, rotation=15)
    # Set descriptions:
    plt.title("Amount of religions per geographical region")
    plt.ylabel('Avg. Amount of Religions per country')
    plt.xlabel('')
    plt.show()



'''
2002
Sikhs 0.38%, other religions 12.6%, 
non-religious 12.63%, atheists 2.47% (2000 est.)

2007
Sikhs 0.39%, 
other religions 12.61%, non-religious 12.03%, atheists 2.36% (2004 est.)

'''
christian = {2002:32.88,2007:33.03,2012:33.39,2018:31.4}
muslim = {2002:19.54,2007:20.12,2012:22.74,2018:23.2}
hindu ={2002:13.34,2007:13.34,2012:13.8,2018:15}
buddhist ={2002:5.92,2007:5.89,2012:6.77,2018:7.1}
jewish ={2002:0.24,2007:0.23,2012:0.22,2018:0.2}
other ={2002:13,2007:13,2012:12.3,2018:6.7}
unaffiliated = {2002:15.01,2007:14.39,2012:11.6,2018:16.4}
ls = [christian, muslim, hindu,buddhist,jewish,other,unaffiliated]
names = ["christian", "muslim", "hindu", "buddhist", "jewish", "other", "unaffiliated"]

x = list(christian.keys())
for rel, name in zip(ls, names):
    y = list(rel.values())
    print(name)
    plt.plot(x, y, label=name, linewidth=3)
plt.title("Major Religions Global Trend", fontsize=16)
plt.ylabel("% of global population", fontsize=14)
plt.legend(loc=2, fontsize=14)
plt.show()