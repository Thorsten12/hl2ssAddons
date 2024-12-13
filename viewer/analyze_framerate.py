import csv
from datetime import datetime
import statistics
import os

def analyze_framerate(csv_filename):
    timestamps = []
    
    # Open the CSV file and parse the data
    with open(csv_filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=';')  # Use semicolon as the delimiter
        next(csv_reader)  # Skip the header row
        
        for row in csv_reader:
            if row:  # Ensure the row is not empty
                try:
                    # Extract and parse the timestamp (first column)
                    timestamp = datetime.fromisoformat(row[0].strip())
                    timestamps.append(timestamp)
                except ValueError:
                    print(f"Skipping invalid timestamp: {row[0]}")
    
    if len(timestamps) < 2:
        print("Not enough data points to calculate frame rate.")
        return
    
    # Calculate time differences between frames
    time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
    time_diffs = [diff for diff in time_diffs if diff > 0]  # Remove zero or negative differences
    
    if not time_diffs:
        print("All time differences are zero or invalid. Unable to calculate frame rate.")
        return
    
    total_time = timestamps[-1] - timestamps[0]
    average_diff = statistics.mean(time_diffs)
    framerate = 1 / average_diff if average_diff > 0 else 0
    
    min_framerate = 1 / max(time_diffs) if max(time_diffs) > 0 else 0
    max_framerate = 1 / min(time_diffs) if min(time_diffs) > 0 else 0
    
    # Display results
    print(f"Number of frames: {len(timestamps)}")
    print(f"Average frame rate: {framerate:.2f} FPS")
    print(f"Minimum frame rate: {min_framerate:.2f} FPS")
    print(f"Maximum frame rate: {max_framerate:.2f} FPS")
    print(f"Standard deviation of frame intervals: {statistics.stdev(time_diffs):.4f} seconds")
    print(f"Total recording time: {total_time}")

def find_csv_file_in_directory(directory):
    # Look for the first CSV file in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            return os.path.join(directory, filename)
    return None

# Prompt user for directory containing the CSV file
input_directory = input("Enter the directory containing the CSV file: ")

# Locate the CSV file in the specified directory
csv_filename = find_csv_file_in_directory(input_directory)

if csv_filename:
    print(f"Analyzing file: {csv_filename}")
    analyze_framerate(csv_filename)
else:
    print("No CSV file found in the directory.")