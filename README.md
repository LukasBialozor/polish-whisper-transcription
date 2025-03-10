# Polski Transkryptor Audio z Whisper

Ten projekt wykorzystuje model OpenAI Whisper do transkrypcji nagrań audio w języku polskim. Program działa w środowisku Jupyter Notebook i oferuje zarówno transkrypcję, jak i podstawową analizę tekstu.

## Funkcje

- Transkrypcja plików audio w języku polskim
- Opcjonalne tłumaczenie na język angielski
- Analiza tekstu transkrypcji:
  - Statystyki tekstu (liczba słów, zdań, unikalnych wyrazów)
  - Segmentacja czasowa (określanie momentów początku i końca wypowiedzi)
  - Prosta analiza sentymentu/tonacji
  - Wyszukiwanie słów kluczowych
- Generowanie wykresów i wizualizacji
- Zapisywanie wyników w formatach TXT i JSON

## Wymagania

Do uruchomienia projektu potrzebujesz Pythona 3.8+ oraz kilku bibliotek. Wszystkie wymagane pakiety znajdziesz w pliku `requirements.txt`.

### Instalacja zależności

```bash
pip install -r requirements.txt
```

### Model Whisper

Program wykorzystuje model OpenAI Whisper, który zostanie automatycznie pobrany przy pierwszym uruchomieniu. Dostępne modele:

- **tiny** (~75 MB) - najszybszy, najmniej dokładny
- **base** (~150 MB) - szybki, dobra jakość
- **small** (~500 MB) - dobry kompromis 
- **medium** (~1.5 GB) - bardzo dobra jakość
- **large** (~3 GB) - najlepsza jakość
- **large-v3** (~3 GB) - najnowsza wersja, najlepsza jakość

W kodzie używamy modelu `large-v3`, ale możesz to zmienić według potrzeb.

## Jak używać

1. Zainstaluj wszystkie zależności
2. Uruchom Jupyter Notebook: `jupyter notebook`
3. Otwórz plik `whisper_transcription.ipynb`
4. Zmień ścieżkę do pliku audio (zmienna `audio_path`)
5. Uruchom komórki notebooka

## Uwagi

- Transkrypcja używa biblioteki `librosa` zamiast FFmpeg, więc nie ma potrzeby instalowania dodatkowego oprogramowania.
- Pierwsze uruchomienie pobiera model, co może zająć kilka minut w zależności od jego rozmiaru.
- Dla dłuższych nagrań (>10 minut) zalecane jest użycie modelu `base` lub `small` do szybszej transkrypcji.
- Model działa offline - po pobraniu modelu nie wymaga połączenia z internetem.

## Przykład użycia

```python
# Załaduj model
model = whisper.load_model("large-v3")

# Transkrybuj plik audio
result = transcribe_audio("nagranie.mp3", language="pl")

# Wyświetl transkrypcję
print(result["text"])
```

## Dostosowanie

Możesz dostosować skrypt do swoich potrzeb:

- Zmień model na mniejszy (`base` lub `small`) dla szybszej transkrypcji
- Dostosuj listę słów kluczowych do wyszukiwania
- Rozbuduj listę słów do analizy sentymentu
- Dodaj własne funkcje analizy tekstu
