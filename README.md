# Analiza/Predykcja Cen Samochodowych

## Opis Dokumentu

Ten dokument to dokumentacja analizy i predykcji cen samochodowych. Analiza dotyczy zbioru “sales_ads_train.csv” przygotowanego przez PJATK Data Science Club (DSC). W ramach projektu zmagamy się z problemem ściśle regresyjnym polegającym na określeniu najdokładniejszej ceny samochodu przy pomocy 23 unikalnych atrybutów wymienionych w sekcji **Opis zbioru danych**. Dla tego problemu i czyszczenia danych dobrany został program dataiku w celu zapoznania się z nim i przetestowania jak działa.

## Opis Zbioru Danych

Zbór danych dzieli się na 4 pliki `.csv`:

- `Sales_ads_train`
- `Sales_ads_test`
- `Synthetic_training_data_mostlyai_pl.csv`
- `synthetic_training_data_sdv_pl.csv`

Każdy ze zbiorów zawiera 25 kolumn:

- **ID** – unikalny identyfikator ogłoszenia
- **Cena** – cena pojazdu (Atrybut decyzyjny, target value)
- **Waluta** – waluta ceny (głównie polski złoty, ale również euro)
- **Stan** – stan pojazdu (nowy lub używany)
- **Marka Pojazdu** – marka pojazdu
- **Model Pojazdu** – model pojazdu
- **Generacja Pojazdu** – generacja pojazdu
- **Wersja Pojazdu** – wersja pojazdu
- **Rok Produkcji** – rok produkcji samochodu
- **Przebieg Km** – przebieg w kilometrach
- **Moc KM** – moc silnika w koniach mechanicznych
- **Pojemność Cm3** – pojemność silnika w centymetrach sześciennych
- **Rodzaj Paliwa** – rodzaj paliwa
- **Emisja CO2** – emisja CO₂ w g/km
- **Napęd** – rodzaj napędu
- **Skrzynia Biegów** – typ skrzyni biegów
- **Typ Nadwozia** – typ nadwozia
- **Liczba Drzwi** – liczba drzwi
- **Kolor** – kolor nadwozia
- **Kraj Pochodzenia** – kraj pochodzenia pojazdu
- **Pierwszy Właściciel** – czy właściciel jest pierwszym właścicielem
- **Data Pierwszej Rejestracji** – data pierwszej rejestracji pojazdu
- **Data Publikacji Oferty** – data publikacji ogłoszenia
- **Lokalizacja Oferty** – lokalizacja podana przez sprzedającego
- **Wyposażenie** – lista wyposażenia pojazdu (np. ABS, poduszki powietrzne, czujniki parkowania itp.)

## Analiza Wstępna

W przypadku regresji, czyli estymacji wartości liczbowej z dziedziny liczb rzeczywistych, należy zwrócić uwagę na pułapki kryjące się w zbiorze danych. Zbór `sales_ads_train.csv` zawiera 135k rekordów po 25 kolumn każdy (~500MB). Przetwarzanie całego zbioru może być kosztowne.

### Kluczowe Pytania

- Czy nie lepiej wziąć pod uwagę mniejszą próbkę zbioru danych?

### Obsługa Pustych Danych

Możliwe opcje:

1. **Usunięcie wierszy** zawierających wartości puste.
2. **Usunięcie kolumn** zawierających wartości puste.
3. **Uzupełnienie wartości** pustych na podstawie reszty danych.
4. **Flagowanie** pustych wartości jako "nieznane".

### Ocena Metod

- Usuwanie wierszy zmniejsza zbór danych.
- Usuwanie kolumn pozbawia istotnych informacji.
- Uzupełnianie średnią/modą jest skuteczne dla danych ciągłych.
- Flagowanie dobrze działa na danych kategorycznych.
- Algorytmiczne uzupełnianie (Random Forest) daje lepsze wyniki.

## Analiza Korelacji

Warto szukać zmiennych skorelowanych z ceną. Macierz korelacji pokazuje:

- Silna korelacja między **Rokiem Produkcji** a **Ceną**.
- Wysoka korelacja między **Mocą** i **Pojemnością**.

## Jak Pozbyć Się Brakujących Wartości?

- **Usuwanie rekordów**: Pozostało tylko 4_000 z 135_000 rekordów.
- **Algorytmiczne uzupełnianie** (Random Forest) pozostawiło 115_000 rekordów.
- **Dokładność przewidywania marki**: 70%.

## Trenowanie i Testy Modeli

Zbór treningowy podzielono na podzbiory testowy i treningowy.

- **RMSE modelu bazowego**: 70_000.
- **RMSE Random Forest**: 50_000.

## Inne Rozwiązania

### Potencjalne Kierunki Dalszych Działań

1. **Scalanie Zbiorów Danych**: 
   - Połączenie zbiorów oryginalnych i uzupełnionych.
2. **Standardowy Pipeline dla Zadania Regresji**:

### Etapy Przetwarzania Danych

- Konwersja cen do jednej waluty.
- Ekstrakcja wieku pojazdu.
- Kodowanie danych kategorycznych.
- Usunięcie zbędnych kolumn.
- Uzupełnianie braków flagami.

### Testowanie Modeli

- **Random Forest for Regression**
- **XGBoost**
- **LightGBM**
- **ExtraTree**

### Ewaluacja Wyników

- Optymalizacja hiperparametrów (Optuna framework).
- Miary sukcesu: **RMSE**, **MAE**, **R²**.
- Dodatkowa poprawa RMSE o **1 000**.

---
