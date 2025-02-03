import requests
import pandas as pd

import csv
import re

def load_txt_file():
    # URL of the file
    url = "https://myweb.cmu.ac.th/sansanee.a/NNGr/dataset/Flood_dataset.txt"

    try:
        # Send GET request to download the file
        response = requests.get(url)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Read the content
        content = response.text
        
        # Option 1: Print the content
        print(content)
        
        # Option 2: Save the content to a local file
        with open("flood_dataset.txt", "w", encoding="utf-8") as file:
            file.write(content)
        print("File has been successfully downloaded and saved as 'flood_dataset.txt'")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def pat_to_csv(pat_filename, csv_filename):
    # อ่านไฟล์ .pat
    # อ่านไฟล์แบบพื้นฐาน
    with open(pat_filename, 'r') as file:
        # ข้ามบรรทัดแรกที่เป็นหัวตาราง
        header = file.readline()
        station = ['s1','s2','p']
        header = [station[j//4]+h for j,h in enumerate(header.strip().split())]
        print(header)
        # อ่านข้อมูลทีละบรรทัด
        data = []
        for i,line in enumerate(file):
            values = [float(x) for x in line.strip().split()]
            data.append(values)
    
    # เขียนข้อมูลลงไฟล์ CSV
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

# ตัวอย่างการใช้งาน
pat_filename = 'flood_dataset.txt'
csv_filename = 'flood.csv'

try:
    pat_to_csv(pat_filename, csv_filename)
    print(f"แปลงไฟล์สำเร็จ: {csv_filename} ถูกสร้างขึ้น")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {str(e)}")