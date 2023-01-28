# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?

**Antwoord**

De architectuur die is uitgekozen is een eenvoudig model. Door de lineaire laders i.c.m. de activation functions is het model is staat om te leren. Het nadeel van dit model is dat het niet is toegespitst op de aard van het probleem. De onderdelen die er op dit moment staan zijn wel belangrijke bouwstenen in het uiteindelijke model. Voor dit specifieke probleem is de huidige architectuur niet geschikt, omdat het time series betreft. Er kan beter gekozen worden om RNN/GRU/LSTM layers toe te voegen en deze vervolgens aan te laten sluiten op de gekozen lineaire structuur. 

Het aantal layers is lastig om zo wat over te zeggen. Bij te weinig layers bestaat de kans van underfitting en bij teveel layers bestaat de kans van overfitting. 2 hidden layers lijkt een goede start.

Hieronder de voor- en nadelen van een lineair neuraal netwerk:

**Sterke punten**
* Een lineair neuraal netwerk kan met relatief weinig data een voorspelling maken. De dataset met 8800 datapunten is aan de krappe kant voor deep learning.
* Het model kan gebruikt worden als een baseline. Een lineair model kan daardoor dienen als baseline om te testen of de nieuwe architectuur beter presteert.

**Zwakke kanten**
* Dataset is gericht op time series en heeft daardoor 3 dimensies. Een lineair model heeft 2 dimensies als input. Een GRU/LSTM/RNN is geschikter.




- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?

**Antwoord**

**Units**

De input layer staat vast met 13 features. Dit zijn de features uit de dataset per gesproken getal. Zoals het nu in de config staat is het goed.

Er is geen wetmatigheid wat betreft het aantal units en is afhankelijk van de dataset. Vaak komt dit neer op trail & error om een goede richting te bepalen. Een eerste hidden layer van 100 units kan goed zijn om uiteindelijk 20 klasses te voorspellen.

Interessanter is de 2e hidden layer met 10 units. Dit is kleiner dan het aantal units in de output layer (20). Dit betekent dat het model elementen moet verzinnen om bij het aantal units in de output layer te komen. Mijn advies zou zijn om als eerste te testen met een groter aantal units in de 2e hidden layer dan het aantal in de output layer (20).

Het doel is om 20 klasses te voorspellen in een classificatiemodel: vrouw, man, 10 cijfers. De output is, zoals het ook in de config staat 20.  Hier hoeft niets aan te veranderen.

*Na een eerste test*
Door de 2e hidden layer aan te passen naar 50 is het model in staat om met 75% nauwkeurigheid een voorspelling te doen. Een grote verbetering t.o.v. 67%.

**Dropout**

Drop-out is een goede methode om overfitting tegen te gaan. In feite worden features weggelaten in de trainset. Dit zorgt ervoor dat het model de features niet kan memoriseren. 

Het nadeel van dropout is informatieverlies. Als er niet veel data beschikbaar is kan dit ten koste gaan van de nauwkeurigheid. Tevens is het geval dat volgordelijkheid in time series belangrijk is voor de uitkomst. Door een hoge dropout gaat dit ten koste van de volgordelijkheid.

Een dropt van 0.5 betekent dat 50% van de data wordt weggegooid tijdens het trainen. Met een dataset van 8800 records en 13 features per record is 50% achterwege laten aan de hoge kant. Mijn advies zou zijn om te starten met een dropout van 0.2. 

Als bestanden heel groot zijn kan er een hogere dropout gehanteerd worden.





## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)

**Antwoord**

De dataset leent zich voor een time series model. Kenmerkend voor een time series model is 3-dimensionaliteit. Een lineair model is 2-dimensionaal.  Het probleem dat de junior probeert op te lossen is het aantal dimensies aan te laten sluiten bij het gekozen lineaire model. 

Door *x = x.mean(dim=1)* te gebruiken wordt het gemiddelde van de eerste dimensie genomen, waardoor de tensor 2-dimensionaal wordt ipv 3-dimensionaal. Op deze manier is de data geschikt voor het lineaire model.




- Hoe had hij dit ook kunnen oplossen?

**Antwoord**

Door *flatten* te gebruiken is het ook mogelijk om van 3 naar 2 dimensies te gaan. Het converteert een matrix naar een enkele rij. Flatten kan gebruikt worden tussen de GRU/LSTM lagen en de lineaire lagen als een soort koppelstuk.





- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?

**Antwoord**

**Voordeel mean**

Minder rekenkracht benodigd, omdat er 1 feature wordt meegenomen. Hierdoor werkt het model sneller.

**Nadeel mean**

Verlies van informatie, omdat er maar 1 feature wordt meegenomen.
Verlies van volgordelijkheid, omdat een gemiddelde wordt genomen. In een time series is het van groot belang dat de volgordelijkheid intact blijft.

**Voordeel flatten**

Flatten behoudt informatie van de stappen. De features worden als het ware achter elkaar gezet. Hierdoor gaat veel minder informatie verloren.

**Nadeel flatten**

Meer rekenkracht benodigd, omdat er meer features zijn.

### Conclusie
In dit geval zou ik kiezen voor flatten, omdat het gaat om een time series model. Volgordelijkheid is belangrijk, omdat er per stap nieuwe informatie wordt doorgegeven. *Mean* zou voor mij geen optie zijn voor dit model. *last timestep* zou voor mij wel het overwegen waard zijn, omdat het hidden state heeft meegekregen van de voorgaande reeks.






### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.

**Antwoord**

Deze casus betreft een classificatieprobleem met time series. Kenmerkend voor time series is de 3-dimensionaliteit (batch, channel, timesteps). Bij time series is er de keuze uit de volgende layers: RNN, GRU en LSTM. Ook Attention heeft 3 dimensies, maar is beter geschikt voor NLP. Daar is deze casus niet op gericht.

**RNN, GRU, LSTIM**

De keuze voor deze layers is in te beantwoorden met de volgende vragen:

* Zorgt geheugen in het model voor betere prestaties?
* Hoe complex is de dataset?

Als er geen geheugen benodigd is, dan volstaat een RNN model. Presteert het model beter met geheugen en is de dataset niet complex, dan is GRU waarschijnlijk de beste keuze. Bij een complexe dataset en benodigd geheugen kan LSTM uitkomst bieden.

Geheugen wordt interessant als het meer tijdstappen betreft, bijvoorbeeld meer dan 10 tijdstappen. De ingesproken getallen bestaan uit 13 tijdstappen, waardoor geheugen in het model een interessante optie is.

13 tijdstappen, 8800 records en 20 klasses lijkt op het oog niet een hele complexe en grote dataset. Met dit in ogenschouw zouden mijn bovenste layers, GRU layers worden. Na een aantal GRU-layers zou ik flatten gebruiken om over te gaan naar linear-layers om te classificeren.



- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.

**Antwoord**

Mijn config voor dit model zou als volgt zijn:

* input_size: 13
* hidden_size: [128, 32]
* dropout: [0.1, 0.2]
* num_layers: [3, 4]
* output_size: 20

Een ander probleem waarin GRU werd toegepast is *gestures* van polsbewegingen. Hierin werden polsbewegingen geclassificeerd in 20 bewegingen en was dit een geschikte configuratie. Bij een te grote *hidden_size* kost het veel rekenkracht en is er kans op overfitting. Bij een te lage *hidden_size* is het model aan het underfitten op de data en mist het belangrijke elementen om tot een goede classificatie te komen.

Een grote range kan ervoor zorgen dat je zoekt naar een spelt in een hooiberg. Door eerst manueel testen uit te voeren geeft dit een goede indicatie wat de range kan zijn. Manueel *sizes* testen kan gedaan worden door met een laag (3) aantal epochs een range van getallen te testen, bijvoorbeeld 32, 64, 128 en 256. De beste resultaten kan een range opleveren.





- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).

**Antwoord**

Voor mij zou de architectuur er als volgt uit komen te zien.

```
self.rnn = nn.GRU(
            input_size=config["13"],
            hidden_size=config["64"],
            dropout=config["0.2"],
            batch_first=True,
            num_layers=config["3"],
        )
        self.linear = nn.Linear(config["64"], config["20"])
```

De input (13) en output (20) staan vast. Er zijn vaak meer hidden features nodig dan output size, vandaar de keuze voor 64. Dan zijn er afgerond 3 features per klasse. Afgaande op eerdere datasets zoals *gestures* kan het model een hoge accuraatheid benaderen. De dropout wordt op 0.2 gezet, omdat de dataset niet heel groot is. Het aantal lagen is nog lastig te bepalen. Het lijkt niet een ingewikkelde dataset, vandaar de keuze voor 3 lagen.





### 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
- Maak in `settings.py` een nieuwe config voor jouw model
- Train het model met enkele educated guesses van parameters. 
- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.
- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.

Hieronder een voorbeeld hoe je een plaatje met caption zou kunnen invoegen.

<figure>
  <p align = "center">
    <img src="img/motivational.png" style="width:50%">
    <figcaption align="center">
      <b> Fig 1.Een motivational poster voor studenten Machine Learning (Stable Diffusion)</b>
    </figcaption>
  </p>
</figure>

**Antwoord**

De werking van het model ga ik in kaart brengen door 3 experimenten uit te voeren met verschillende parametersets. Onderdelen van de parametersets die verandert worden zijn: *hidden_size*, *dropout*, *num_layers*.

### Experiment 1
Dit is de architectuur die ik op voorhand bedacht had in de voorgaande vraag.

* Hidden_size = 64
* Dropout = 0.2
* Num_layers = 3

Hieronder volgen de uitkomsten van het eerste experiment.
![](Tentamen%20ML22/Experiment%201.png)

Het experiment is ingezet met 50 epochs. Uit de data blijkt dat na 20 epochs het model bijna op z’n top getraind is (0.96 accuracy). De grafiek van test- en trainset blijven geleidelijk aflopen. Er is daardoor geen sprake van overfitting. Dit betekent dat de dropout niet hoger gezet hoeft te worden.

### Experiment 2
In het 2e experiment heb ik de dropout toch hoger gezet, omdat dit de initiële instelling was. Ik wil verifiëren of deze instelling daadwerkelijk de beste oplossing is voor dit probleem. Vervolgens wil ik het vergelijken met mijn eigen oplossing.

* Hidden_size = 64
* Dropout = 0.5
* Num_layers = 3

![](Tentamen%20ML22/Experiment%202.png)

Met deze instellingen gebeurt er iets bijzonders op het eind en lijkt voornamelijk impact te hebben op de testset. Ik vind het lastig om te verklaren waarom dit gebeurt, maar het zou te maken kunnen hebben dat er dusdanig veel informatie is verloren, dat er een grote foutmarge is. Er is ook een drop in accuracy zichtbaar. Op de top was accuracy 93 en daarmee minder accuraat dan een dropout van 0.2.

### Experiment 3
Hogere *hidden_size* betekent meer informatie. Ook heb ik het aantal layers vergroot naar 5. Ik ben benieuwd wat er gebeurt als het model groter wordt en of het de extra rekenkracht waard is.

* Hidden_size = 128
* Dropout = 0.5
* Num_layers = 5

![](Tentamen%20ML22/Experiment%203.png)

Afgaande op deze grafieken is de lijn grilliger dan bij het eerste experiment. De trainset gaat lekker en blijft smooth, terwijl de grafiek op de testset grillig is en af en toe ook naar boven schiet. Dit kan erop duiden dat het model kwetsbaar is voor overfitting. Wellicht dat een hogere dropout hier uitkomst kan bieden. De accuracy was tevens 0.96, vergelijkbaar met het eerste experiment.

### Conclusie
Het eerste experiment lijkt het meest succesvol, omdat geen teken van overfitting vertoont en de accuracy het hoogst is. Rondom deze parameters is voor mij het startpunt voor hypertuning.




## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- Stel dat je een model maakt of een dataloader aanpast. Maak dan een nieuw model, zodat duidelijk is welke aanpassingen in de code heb gedaan.
- voeg jouw model in op de juiste plek in de `tune.py` file.
- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.
- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.


### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.
- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

**Antwoord vraag 2a, 2b, 2c**
### Hypertune V1.0
Als eerst heb ik een searchspace aangemaakt voor mijn GRU-model. Met de architectuur vanuit 1D is een nauwkeurigheid behaald van 96% en zie geen reden voor aanpassing aan het model. De gekozen parameters liggen rond de parameterset uit 1D, met daarin de beschikbare resources in ogenschouw genomen.

* Hidden_size = 64, 256. Dit is nog een relatief groot zoekgebied. In eerdere testen bleek tussen deze range de hoogste accuracy behaald werd.
* Num_layers = 2, 4. Minimaal 2 lagen en maximaal 4. Bij teveel lagen kan overfitting voorkomen.
* Dropout = 0.2, 0.4. Dataset is niet heel groot, vandaar een kleinere dropout dan 0.5. De ondergrens van 0.2 gekozen, omdat dit vaak als standaard gehanteerd wordt.
* Batchsize = 80, 600. Erg breed, maar gekozen om een indicatie te krijgen.

Hieronder de resultaten van de eerste hypertune.

![](Tentamen%20ML22/63E02EE5-24D2-43CB-ADB2-B8EF09101871.png)

![](Tentamen%20ML22/DDF93EF8-E8A3-428C-8BD0-53AC0896F54C.png)

![](Tentamen%20ML22/A686FB88-2124-4849-B3C2-91231BA04B19.png)

Een accuracy van 94% is nog niet de 96% van de handmatige tuning, dus er is nog ruimte voor verbetering. Interessant uit deze uitkomst is dat de *hidden_size* (238) groter is dan de handmatige tuning en het aantal *layers* kleiner (2). De *loss* ziet er goed uit voor de test- en trainset en vertoont geen tekenen van overfitting. Kortom een goede basis om een gerichter experiment uit te voeren. Met name de *batchsize* is erg groot (80, 600) voor deze dataset. *Batchsize* wordt verkleind en het aantal epochs wordt opgeschaald naar 50.

### Hypertune V2.0
In navolging op het voorgaande experiment zijn de ranges in de *searchspace* verkleind en is er voortborduurt op het beste experiment in Hypertune 1.0. Ook is de *batchsize* verkleind in verband met de grootte van de dataset en stelt de traanloop in staat om meer epochs te trainen.

* Hidden_size = 220, 256. Dit is nog een relatief groot zoekgebied. In eerdere testen bleek tussen deze range de hoogste accuracy behaald werd.
* Num_layers = 2, 4. Minimaal 2 lagen en maximaal 4. Bij teveel lagen kan overfitting voorkomen.
* Dropout = 0.2, 0.3. Dataset is niet heel groot, vandaar een kleinere dropout dan 0.5. De ondergrens van 0.2 gekozen, omdat dit vaak als standaard gehanteerd wordt.
* Batchsize = 80, 100. In de vorige tune kwam 83 als best eruit. Dit ligt in lijn met de verwachting dat de batchsize verkleind moest worden.

![](Tentamen%20ML22/FF9D3FA6-4E67-44AE-A3BC-8023433C8383.png)


![](Tentamen%20ML22/4D794CAA-4CF7-4140-8B8A-113CCE4ED94B.png)

Bovenstaand de resultaten van de Hypertune 2.0. Hieruit blijkt dat de *hidden_size* weer dichtbij het voorgaande experiment ligt, ditmaal 233. De optimale range lijkt in *hidden_size* gevonden. Opvallend is dat er een layer is bijgevoegd, van 2 naar 3. De dropout ligt dichtbij de 2. Het is het proberen waard om vanaf 0.1 tot 0.21 te testen.

![](Tentamen%20ML22/6E928F5D-258F-4C2F-A117-1926A62AB7DF.png)

De *loss* voor de train en test-set zien er goed uit en geven wederom geen teken van overfitting, ondanks het toevoegen van een extra layer.

### Hypertune V3.0
Dit is de laatste setup die getest wordt. De searchspace wordt nog verder verkleind door de input van eerdere setups in 1.0 en 2.0. De volgende setup wordt getest:

* Hidden_size = 230, 240. 10 opties om de *hidden_size* nog verder te optimaliseren.
* Num_layers = 2, 4. Minimaal 2 lagen en maximaal 4. In de voorgaande experimenten kwamen 2 en 3 layers als beste naar voren. Vandaar geen aanpassing hierin.
* Dropout = 0.1, 0.21. Zoals aangegeven in voorgaande hypertune lag dropout aan de onderkant van de range 0.2 tot 0.3, vandaar de aanpassing naar 0.1 tot 0.21.
* Batchsize = 85, 95. In de vorige tunes kwamen 83 (V1.0) en 93 (V2.0) naar voren als beste *batchsize*.

![](Tentamen%20ML22/0C80751A-B0DE-4B9F-ADA1-ECAE26EFBAB5.png)

![](Tentamen%20ML22/186C7C3A-30B1-4053-AF3D-EBA8C822A007.png)

Een lagere dropout leidt niet tot betere resultaten. Het voorgaande experiment was succesvoller dan deze. Toch komt deze test tot een accuracy van 95%.

![](Tentamen%20ML22/CF54030B-4892-49CC-83B7-4ADD79B58581.png)

*Loss* in train- en testset geven geen teken van grilligheid of overfitting.

### Conclusie
Met hypertune 2.0 komt een accuracy van 97% uit het model. Dit lijkt na een aantal experimenten het maximale van het GRU model. Wellicht dat een ander model, zoals AttentionGRU een beter resultaat kan halen. Dit is tot nu toe de prijswinnende parameterset voor hypertuning.

```
class gru_modelSearchSpace(BaseSearchSpace):
    hidden_size: Union[int, SAMPLE_INT] = tune.randint(220, 256)
    num_layers: Union[int, SAMPLE_INT] = tune.randint(2, 4)
    dropout: Union[float, SAMPLE_FLOAT] = tune.uniform(0.2, 0.3)
    batchsize: Union[int, SAMPLE_INT] = tune.randint(80, 100)
```

## Vraag 3
### 3a
- fork deze repository.
- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.
- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages
- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
