import os
import csv
from datetime import datetime
import statistics

def analyze_framerate(csv_filename):
    timestamps = []
    
    # Öffnen der CSV-Datei und Lesen der Daten
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Überspringen der Kopfzeile
        
        for row in csv_reader:
            if row:  # Überprüfen, ob die Zeile nicht leer ist
                timestamp = datetime.fromisoformat(row[0])
                timestamps.append(timestamp)
    
    if len(timestamps) < 2:
        print("Not enough data points to calculate frame rate.")
        return
    
    # Berechnung der Zeitdifferenzen zwischen den Frames
    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
    time_diffs = [diff for diff in time_diffs if diff > 0]  # Entferne Zeitdifferenzen von 0
    
    if not time_diffs:
        print("All time differences are zero. Unable to calculate frame rate.")
        return
    
    all_time = timestamps[-1] - timestamps[0]
    average_diff = statistics.mean(time_diffs)
    framerate = 1 / average_diff
    
    min_framerate = 1 / max(time_diffs)
    max_framerate = 1 / min(time_diffs)
    
    # Ergebnisse anzeigen
    print(f"Number of frames: {len(timestamps)}")
    print(f"Average frame rate: {framerate:.2f} FPS")
    print(f"Minimum frame rate: {min_framerate:.2f} FPS")
    print(f"Maximum frame rate: {max_framerate:.2f} FPS")
    print(f"Standard deviation of frame intervals: {statistics.stdev(time_diffs):.4f} seconds")
    print(f"Total recording time: {all_time}")

def find_csv_file_in_directory(directory):
    # Sucht nach der ersten CSV-Datei im angegebenen Verzeichnis
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            return os.path.join(directory, filename)
    return None

# Ordner vom Benutzer abfragen
input_directory = input("Enter the directory containing the CSV file: ")

# CSV-Datei im angegebenen Ordner suchen
csv_filename = find_csv_file_in_directory(input_directory)

if csv_filename:
    print(f"Analyzing file: {csv_filename}")
    analyze_framerate(csv_filename)
else:
    print("No CSV file found in the directory.")
