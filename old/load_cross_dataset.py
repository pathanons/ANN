import csv
import re

def pat_to_csv(pat_filename, csv_filename):
    # อ่านไฟล์ .pat
    with open(pat_filename, 'r') as file:
        content = file.read()
    
    # แยกข้อมูลแต่ละจุด
    # ใช้ regex เพื่อจับคู่รูปแบบข้อมูล
    pattern = r'p(\d+)\n([\d.]+)\s+([\d.]+)\n(\d)\s+(\d)'
    matches = re.finditer(pattern, content)
    
    # เตรียมข้อมูลสำหรับเขียนลง CSV
    data = []
    headers = ['x', 'y', 'class_1','class_2']
    
    for match in matches:
        point_num = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        class_0 = int(match.group(4))
        class_1 = int(match.group(5))
        
        data.append([ x, y, class_0, class_1])
    
    # เขียนข้อมูลลงไฟล์ CSV
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)

# ตัวอย่างการใช้งาน
pat_filename = 'cross.pat'
csv_filename = 'cross.csv'

try:
    pat_to_csv(pat_filename, csv_filename)
    print(f"แปลงไฟล์สำเร็จ: {csv_filename} ถูกสร้างขึ้น")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {str(e)}")

# ตัวอย่างการใช้งาน
pat_filename = 'ellipse.pat'
csv_filename = 'ellipse.csv'

try:
    pat_to_csv(pat_filename, csv_filename)
    print(f"แปลงไฟล์สำเร็จ: {csv_filename} ถูกสร้างขึ้น")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {str(e)}")