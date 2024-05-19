# Machine Learning kurzus EDA checkpoint

A mi projektünk egy olyan webapplikáció, ahova a felhasználó feltölti a saját, keresztmetszeti adatát, amin prediktálni szeretne, bele tud nézni az adatába, majd modellt és paramétereket is választhat. Jelenlegi verzióban még csak az EDA funkciók szerepelnek, ezek sem feltétlenül véglegesen. 

- A Homepage fülön lehet adatot feltölteni és az egyes változókat kategórikussá, vagy numerikussá alakítani
- A Pandas profiling fülön a teljes adatra elkészíti a pandas profiling-ot (ami egy előre implementált leíró-statisztikákat készító package)
- Az EDA oldalon pedig a felhasználó maga készíthet leíró statokat:
    - Descriptive Statistics: Numerikus változók eloszlása táblázatosan
    - Value counts: Kategórikus változók bar-plotja. Erre azért van szükség, mert a pandas profiling a kategórikus változókból (amik jellemzően string-ek) szófelhőt csinál, ami nem túl informatív. Azokat a változókat lehet itt plotolni, amelyeket a felhasználó a homepage-en kategórikussá állít.
    - Association figures: Két kiválasztott változó kapcsolatát ábrázolja (változótípusok határozzák meg az ábra típusát)
    - Correlation matrix: Összes numerikus változó közötti korrelációs mátrixa
    - Missing values: Hiányzó értékek ábrázolása minden változóra. (A csatolt adatokban mondjuk pont nincsenek hiányzó értékek).




A jelenlegi verzió futtatása:
```
pip install -r requirements.txt
streamlit run 1_Homepage.py
```
A requirements telepítés során a ydata-profiling és a streamlit-pandas-profiling néha egy közös dependency miatt ütik egymást, ha ez történik, akkor a requirements telepítés után a manuális ydata-profiling installálás megoldja a problémát.

```
pip install ydata-profiling
```
 A végleges verzióra ezt mindenféleképpen megoldjuk majd, de ez mostanra nem sikerült egyelőre. Nagyobb adaton az ábrák elkészítése még elég lassú, ezt is igyekszünk majd kijavítani a továbbiakban.

 A két próba-adatot [itt](https://drive.google.com/drive/folders/1EjEyW7KaAytb7GVo1A3SJUoad6JzRi8R?usp=sharing) éritek el, mi ezeken teszteltük a működést, de persze mást is fel tudtok tölteni. Az egyik [repülőjáratok késésének előrejelzéséről](https://www.kaggle.com/datasets/threnjen/2019-airline-delays-and-cancellations), a másik pedig [alkoholfogyasztás és dohányzás előrejelzéséről](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset) szól.
