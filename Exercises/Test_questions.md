
Kokchun Giang
1 / 4
Instuderingsfrågor maskininlärning AI21
Facit finns inte till nedanstående frågor. Frågorna ska utgöra ett stöd för er att lära er teori för
maskininlärning. Samarbeta gärna och diskutera gärna med varandra.

1. Vad är data leakage, varför är det dåligt och hur kan du undvika data leakage?  

  
Svar: 

2. Beskriv vad poängen kan vara att dela upp datasetet till träningsdata och testdata för maskininlärning?  

  
Svar: 

3. Beskriv vad poängen kan vara att dela upp datasetet till tränings-, validerings- och testdataset för maskininlärning?  

  
Svar: 

4. Hur skiljer sig batch gradient descent och mini batch gradient descent?  

  
Svar: 

5. Vad är en aktiveringsfunktion i en artificiell neuron? Ge två exempel på aktiveringsfunktioner.  

  
Svar: 

6. Du har ett program som klassificerar email till spam eller hams (inte spam). Vilka evalueringsmetrics ska
du kolla på och varför?  

  
Svar: 

7. Ett brandlarm larmar ofta när det inte brinner. Tror du precision är högt eller lågt, förklara varför.  

  
Svar: 

8. Förklara vad curse of dimensionality är och hur det kan påverka en maskininlärningsmodell.  

  
Svar: 

9. Vad är skillnaden på feature standardization och normalization?  

  
Svar: 

10. Du har ett dataset som ser ut som följande (1000 observationer):

```
temperatur i °C vind (m/s)
22.5    2.0
8.3     10.3
13.2    4
```

... ...  
En meterolog vill gruppera dessa. Ge ett förslag på hur hen skulle kunna göra med en
maskininlärningsalgoritm.  

  
Svar: 
Standardisera. Scala via min max. Så det inte blir ett så stort spann när man plottar.

Vi vill använda en algoritm som grupperar värdena.
SVM kan gruppera. K-means är den mest rimliga på rak arm. 
SVM separerar egentligen bara datan. Den är väldigt känslig för outliers.
Outliers kan vara relevant då tillexempel under en storm kan det ju blåsa över 30m/s och det är extremt ovanligt

Vi har inga labels (target) (facit). 
Så vi behöver en modell som fungerar unsupervised. 

K-means sätter allt i kluster. 
Decision Tree grupperar även datan i olika branches. (tänk baseball exempel)

Jag hade tittat på datan och testat alla tre modeller. 

Sen tror jag att jag hade använt K-Means.
Skapa kluster genom range. 
mellan många olika testa olika K och ta ut Sum Squared dusta

Bestämma K med hjälp av en elbow plot. plotta elbow, Bestäm K genom att ta punkten där det är störst procentuell skillnad mellan två punkter. Välj den sista punkten.

Den punkten har minst distans mellan clustrerna. Viktigast är dock att bestämma tillsammans med en domän expert.

Plotta de olika K och grupperingarna med centroider för domänexpert (Kanske överflödigt att ta med Centroid)

När vi med experten valt K är vi så typ färdig.

Nu kan vi ta med nya punkter och och gruppera dom. 

11. Många maskininlärningsmodeller har svårt att arbeta med textdata direkt. Ge förslag på hur man skulle
kunna förbearbeta en textsträng.  

  
Svar: 

Ta bort stopp ord och punkter, bestämma sig för stora eller små bokstäver.
dela upp texten i individuella ord / tecken.
skala ner ord till grundformen och endelser. så bananer är samma som banan. Eller fötter är fot.
konvertera ord till siffror via one hot encoding. Get dummys.



12. Du ska använda KNN för klassificering. Ge förslag på hur du skulle kunna gå tillväga för att   
ett lämpligt värde på antalet neighbors.  

  
Svar:  
Gå från 1 till n/2.
Skapa elbowplot genom att loopa olika K.
Vi använder Elbowplot för få ett så bra värde som möjligt och spara på processor.
tillexempel ifall k = 10 ger 95% accuracy och k = 100 ger 95.5% så kan det vara värt att ta 10 pga cost / efficency trade off.
Använda Crossvalidation för att få fram rätt svar.


13. Beskriv begreppen: perceptron, multilayered perceptron, feedforward network, fully connected layers.  


  
Svar: 
Perceptron:  Artificiell Neuron i ett neuralt nätverk. Så en nod i nätverket. Tar en input. kör en aktiverings funktion som sedan skickar en output. 
Multilayerd Perceptron: MLP, Som består av flera lager av perceptroner. Lagrerna kallas Input layer, Hidden layer och output layer.  
Feedforward network: Hur datan rör sig genom neutral nätverket. Så från input till output. 
fully connected layers: Alla perceptroner är kopplade till lagret framför. 
Backpropogation: Används när man tränar nätverket. Då kan man gå bakåt i nätverket. 

14. Aktiveringsfunktionen rectified linear unit (ReLU) är definierad som. En nod i ett MLP
har denna aktiveringsfunktion och får inputs och har vikterna och bias  

. Beräkna output för denna nod.  

  
Svar:  
![image.png](../Data/assets/image_1680356948781_0.png)  
Skippa sista steget för Relu.

För att beräkna output y för noden med aktiveringsfunktionen Rectified Linear Unit (ReLU), följer vi dessa steg:

    Beräkna den linjära kombinationen av inputs (x) och vikter (w), och lägg till bias (b).
    Använd ReLU-aktiveringsfunktionen på resultatet från steg 1.

Först, beräkna den linjära kombinationen av inputs och vikter samt bias:

x = (1, 2, 3)ᵀ  
w = (0, 2, 1)ᵀ  
b = -2  

z = xᵀw + b = (1 * 0) + (2 * 2) + (3 * 1) - 2 = 0 + 4 + 3 - 2 = 5

Nu, applicera ReLU-aktiveringsfunktionen:

ReLU(z) = max(0, z) = max(0, 5) = 5

Så output y för noden med ReLU-aktiveringsfunktionen är 5.

15. XOR grind i digitalteknik är en exklusiv disjunktion vilket innebär:

|A |B |A XOR B|
|--|--|--|
| 1 | 1 | 0 |
| 1 | 0 | 1 |
| 0 | 1 | 1 |
| 0 | 0 | 0 |

Visa att följande MLP kan simulera XOR-grind. Varje nod är en perceptron, dvs att aktiveringsfunktionen
är en stegfunktion.  

![image.png](../Data/assets/image_1680160707975_0.png)

  
Svar: 

16. Formeln för Naive Bayes är:  

![image.png](../Data/assets/image_1680160809548_0.png)

Du har ett dataset som består 100 reviews, varav 80 var positiva och 20 negativa. Nedan finns tabeller
på hur frekvent olika ord förekommer för positiva respektive   
Positiva reviews:  

<p style="margin-left: px">


| |happy|good|bookauthor|bad|
|--|--|--|--|--|
|Antal |50 |60 |20 |10 |1|
</p>  

<p style="margin-left: 100px">

Negativa reviews:

| |happy|good|bookauthor|bad|
|--|--|--|--|--|
|Antal |2 |5 |40 |40 |40|
</p>

Vi får ett review med texten "happy bad", använd Naive Bayes och klassificera den till positiv eller
negativ review.  

  
Svar:  
C = Class, k  
Produkt tecken!

![image.png](../Data/assets/image_1680361877405_0.png)
![image.png](../Data/assets/image_1680362063442_0.png)

17. Beskriv kort skillnaderna mellan decision tree och random forest.  

  
Svar:  
Decision Tree tar en datapunkt. Sedan vid varje gren väljer den en av två val och   

Random forest är ett berg med Decision Trees på varandra.
Så datan delas upp mellan flera Decision Trees.
med återläggning.

18. Ge ett exempel på ett maskininlärningsproblem där man kan applicera logistisk regression.  

  
Svar: Prediktera labels.
Spam Ham.

19. Beskriv kort skillnader mellan supervised learning och unsupervised learning.  

  
Svar: 

20. Beskriv kort skillnader mellan regression och klassificering.  

Svar: 

21. Rita upp ett decision tree baserat på följande figur. Tolka därefter detta decision tree.
![image.png](../Data/assets/image_1680161559841_0.png)  
  
Svar: 
  

22. Hitta det logiska felet i följande kod:  
```py
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = df.drop["default"], df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33,
random_state=42)

scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.fit_transform(X_test)

model_SVC = SVC()
model_SVC.fit(scaled_X_train, y_train)

y_pred = model_SVC.predict(scaled_X_test)
# evaluation code ...
```
Svar: 

23. Ett brandlarm har låg precision och hög recall. Diskutera konsekvenserna kring detta brandlarm. På
vilket sätt kan detta vara bra, respektive dåligt?  

Svar: 