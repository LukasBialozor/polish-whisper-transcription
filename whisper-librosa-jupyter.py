# Transkrypcja i analiza nagrań audio w języku polskim
# Używa modelu Whisper i biblioteki librosa (bez potrzeby instalacji FFmpeg)

import os
import numpy as np
import whisper
import json
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import re
import pandas as pd
from datetime import timedelta
import librosa
import warnings

# Ignoruj ostrzeżenia
warnings.filterwarnings("ignore")

# 1. Załadowanie modelu Whisper
# ----------------------------
# Użyj dostępnego modelu (sprawdź nazwę, którą masz zainstalowaną)
model_name = "large-v3"  # Zmień na model, który masz dostępny lokalnie

print(f"Ładowanie modelu Whisper: {model_name}")
try:
    model = whisper.load_model(model_name)
    print(f"Model {model_name} załadowany pomyślnie!")
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {str(e)}")
    print("\nMożliwe przyczyny:")
    print("1. Model nie został jeszcze w pełni pobrany")
    print("2. Podano nieprawidłową nazwę modelu")
    print("3. Model nie został poprawnie zainstalowany w ~/.cache/whisper/")
    print("\nDostępne modele w katalogu ~/.cache/whisper/:")
    cache_dir = os.path.expanduser("~/.cache/whisper/")
    if os.path.exists(cache_dir):
        models = [f.replace(".pt", "") for f in os.listdir(cache_dir) if f.endswith(".pt")]
        for m in models:
            print(f" - {m}")
    else:
        print("Brak katalogu ~/.cache/whisper/")

# 2. Funkcja do transkrypcji audio używająca librosa zamiast FFmpeg
# ----------------------------------------------------------------
def transcribe_audio(audio_path, language="pl", translate=False):
    """
    Transkrybuje plik audio z użyciem modelu Whisper i biblioteki librosa
    
    Args:
        audio_path (str): Ścieżka do pliku audio
        language (str): Kod języka (np. "pl" dla polskiego)
        translate (bool): Czy tłumaczyć na angielski
    
    Returns:
        dict: Wynik transkrypcji
    """
    print(f"Rozpoczynam transkrypcję pliku: {audio_path}")
    
    # Sprawdź, czy plik istnieje
    if not os.path.exists(audio_path):
        print(f"Błąd: Plik {audio_path} nie istnieje!")
        print(f"Pełna ścieżka: {os.path.abspath(audio_path)}")
        return None
    
    # Wybierz odpowiednią opcję zadania
    task = "translate" if translate else "transcribe"
    
    try:
        # Wczytaj plik audio za pomocą librosa zamiast FFmpeg
        print("Wczytywanie pliku audio za pomocą librosa...")
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            print(f"Plik audio wczytany pomyślnie. Długość: {len(audio)/sr:.2f} sekund")
        except Exception as e:
            print(f"Błąd podczas wczytywania pliku audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        # Wykonaj transkrypcję
        print("Rozpoczynam proces transkrypcji...")
        result = model.transcribe(
            audio,
            language=language,
            task=task,
            verbose=True,  # Wyświetlaj informacje o postępie
            fp16=False     # Użyj precision FP32 (bardziej kompatybilne)
        )
        
        print("Transkrypcja zakończona pomyślnie!")
        return result
    except Exception as e:
        print(f"Błąd podczas transkrypcji: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 3. Funkcje do analizy transkrypcji
# ---------------------------------
def analyze_transcription(result):
    """
    Przeprowadza podstawową analizę transkrypcji
    
    Args:
        result (dict): Wynik transkrypcji z Whisper
    
    Returns:
        dict: Statystyki analizy
    """
    if result is None or "text" not in result:
        print("Brak danych do analizy!")
        return None
    
    text = result["text"]
    
    # Podziel na zdania (naiwnie po kropkach, pytajnikach i wykrzyknikach)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Podziel na słowa
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Policz unikalne słowa
    unique_words = set(words)
    
    # Analiza segmentów (jeśli dostępne)
    segments_count = len(result.get("segments", []))
    
    # Podstawowa analiza emocji (bardzo prosta - można rozbudować)
    positive_words = ["dobrze", "świetnie", "super", "doskonale", "wspaniale", "tak", "lubię", "kocham"]
    negative_words = ["źle", "okropnie", "fatalnie", "niestety", "problem", "nie", "trudno", "ciężko"]
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Przygotuj wyniki analizy
    analysis = {
        "total_characters": len(text),
        "total_words": len(words),
        "unique_words": len(unique_words),
        "sentences_count": len(sentences),
        "segments_count": segments_count,
        "average_words_per_sentence": len(words) / max(1, len(sentences)),
        "positive_words_count": positive_count,
        "negative_words_count": negative_count
    }
    
    return analysis

def display_transcription_analysis(result, analysis):
    """
    Wyświetla analizę transkrypcji
    
    Args:
        result (dict): Wynik transkrypcji z Whisper
        analysis (dict): Wynik analizy transkrypcji
    """
    if result is None or analysis is None:
        print("Brak danych do wyświetlenia!")
        return
    
    # Wyświetl pełny tekst transkrypcji
    print("\n" + "=" * 80)
    print("PEŁNA TRANSKRYPCJA:")
    print("=" * 80)
    print(result["text"])
    print("=" * 80)
    
    # Wyświetl podstawowe statystyki
    print("\nPODSTAWOWE STATYSTYKI:")
    print(f"Liczba znaków: {analysis['total_characters']}")
    print(f"Liczba słów: {analysis['total_words']}")
    print(f"Liczba unikalnych słów: {analysis['unique_words']}")
    print(f"Liczba zdań: {analysis['sentences_count']}")
    print(f"Średnia liczba słów w zdaniu: {analysis['average_words_per_sentence']:.2f}")
    
    # Wyświetl wykres tonacji (bardzo uproszczone)
    if analysis['positive_words_count'] > 0 or analysis['negative_words_count'] > 0:
        print("\nPROSTA ANALIZA TONACJI:")
        print(f"Słowa pozytywne: {analysis['positive_words_count']}")
        print(f"Słowa negatywne: {analysis['negative_words_count']}")
        
        # Wykres słupkowy
        plt.figure(figsize=(10, 5))
        plt.bar(['Pozytywne', 'Negatywne'], 
                [analysis['positive_words_count'], analysis['negative_words_count']], 
                color=['green', 'red'])
        plt.title('Analiza tonacji')
        plt.ylabel('Liczba słów')
        plt.show()
    
    # Wyświetl segmenty (przedziały czasowe)
    if 'segments' in result and len(result['segments']) > 0:
        print("\nSEGMENTY TRANSKRYPCJI:")
        segments_df = pd.DataFrame([
            {
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['end'] - segment['start'],
                'text': segment['text']
            }
            for segment in result['segments']
        ])
        
        # Formatuj czas jako czas trwania
        segments_df['start_time'] = segments_df['start'].apply(lambda x: str(timedelta(seconds=round(x, 1))))
        segments_df['end_time'] = segments_df['end'].apply(lambda x: str(timedelta(seconds=round(x, 1))))
        
        # Wyświetl tabelę segmentów
        display(segments_df[['start_time', 'end_time', 'duration', 'text']])
        
        # Wykres długości segmentów
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(segments_df)), segments_df['duration'], color='blue', alpha=0.7)
        plt.xlabel('Numer segmentu')
        plt.ylabel('Czas trwania (s)')
        plt.title('Długość segmentów transkrypcji')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

def search_keywords(result, keywords):
    """
    Wyszukuje słowa kluczowe w transkrypcji
    
    Args:
        result (dict): Wynik transkrypcji z Whisper
        keywords (list): Lista słów kluczowych do wyszukania
    
    Returns:
        dict: Wyniki wyszukiwania
    """
    if result is None or "text" not in result:
        print("Brak danych do wyszukiwania!")
        return None
    
    text = result["text"].lower()
    results = {}
    
    for keyword in keywords:
        count = text.count(keyword.lower())
        results[keyword] = count
    
    # Wyświetl wyniki
    print("\nWYSZUKIWANIE SŁÓW KLUCZOWYCH:")
    for keyword, count in results.items():
        status = "✓" if count > 0 else "✗"
        print(f"{status} {keyword}: {count} wystąpień")
    
    # Wykres słupkowy
    plt.figure(figsize=(12, 6))
    plt.bar(results.keys(), results.values(), color='purple', alpha=0.7)
    plt.xlabel('Słowo kluczowe')
    plt.ylabel('Liczba wystąpień')
    plt.title('Występowanie słów kluczowych')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return results

# 4. Wykonanie transkrypcji na przykładzie
# --------------------------------------
# Poniżej wpisz ścieżkę do swojego pliku audio
audio_path = "1.mp3"  # ZMIEŃ NA WŁAŚCIWĄ ŚCIEŻKĘ! (np. "C:/Users/lbial/1.mp3")

# Próba odtworzenia pliku audio (opcjonalnie)
try:
    print("\nPodgląd pliku audio:")
    display(Audio(audio_path))
except Exception as e:
    print(f"Nie można odtworzyć pliku w notebooku: {str(e)}")
    print("To normalne - można kontynuować transkrypcję.")

# Sprawdzenie ścieżki do pliku
print(f"Pełna ścieżka do pliku: {os.path.abspath(audio_path)}")
if os.path.exists(audio_path):
    print(f"Plik istnieje i ma rozmiar: {os.path.getsize(audio_path)/1024/1024:.2f} MB")
else:
    print(f"UWAGA: Plik nie istnieje pod podaną ścieżką!")

# Wykonaj transkrypcję
result = transcribe_audio(audio_path, language="pl")

# Jeśli transkrypcja się powiodła, przeprowadź analizę
if result:
    # Przeprowadź podstawową analizę
    analysis = analyze_transcription(result)
    
    # Wyświetl wyniki
    display_transcription_analysis(result, analysis)
    
    # Wyszukaj słowa kluczowe (dostosuj do swoich potrzeb)
    keywords = ["przykład", "test", "ważne", "proszę", "dziękuję"]
    search_results = search_keywords(result, keywords)
    
    # Zapisz wyniki do pliku (opcjonalnie)
    output_base = os.path.splitext(audio_path)[0]
    
    # Zapisz transkrypcję jako tekst
    with open(f"{output_base}_transkrypcja.txt", "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    # Zapisz pełny wynik jako JSON
    with open(f"{output_base}_pełny_wynik.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\nTranskrypcja zapisana do: {output_base}_transkrypcja.txt")
    print(f"Pełny wynik zapisany do: {output_base}_pełny_wynik.json")
else:
    print("Nie udało się wykonać transkrypcji. Sprawdź powyższe błędy.")
