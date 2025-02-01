import requests
import pandas as pd

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

import csv
import re

def pat_to_csv(pat_filename, csv_filename):
    # อ่านไฟล์ .pat
    # อ่านไฟล์แบบพื้นฐาน
    with open(pat_filename, 'r') as file:
        # ข้ามบรรทัดแรกที่เป็นหัวตาราง
        header = file.readline()
        
        # อ่านข้อมูลทีละบรรทัด
        data = []
        head = None
        i = 0
        for line in file:
            # แยกข้อมูลด้วยช่องว่างและแปลงเป็นตัวเลข
            if i == 0:
                values = [f's{(idx//4)+1}'+x for idx,x in enumerate(line.strip().split())]
                head = values
            else:
                values = [float(x) for x in line.strip().split()]
                data.append(values)
            i+=1

    # แสดงข้อมูล
    print(head)
    for row in data:
        print(row)
    # # เตรียมข้อมูลสำหรับเขียนลง CSV
    # data = []
    # headers = [
    # 's1t-3',
    # 's1t-2',
    # 's1t-1',
    # 's1t-0',
    # 's2t-3',
    # 's2t-2',
    # 's2t-1',
    # 's2t-0',
    # 't+7']
    # for match in matches:
    #     point_num = match.group(1)
    #     x = float(match.group(2))
    #     y = float(match.group(3))
    #     class_0 = int(match.group(4))
    #     class_1 = int(match.group(5))
        
    #     data.append([ x, y, class_0, class_1])
    
    # # เขียนข้อมูลลงไฟล์ CSV
    # with open(csv_filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(headers)
    #     writer.writerows(data)

# ตัวอย่างการใช้งาน
pat_filename = 'flood_dataset.txt'
csv_filename = 'flood.csv'

try:
    pat_to_csv(pat_filename, csv_filename)
    print(f"แปลงไฟล์สำเร็จ: {csv_filename} ถูกสร้างขึ้น")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {str(e)}")