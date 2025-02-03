import pandas as pd
import numpy as np
def convert_pat_to_csv(input_file, output_file):
    # อ่านไฟล์ .pat โดยใช้ tab เป็นตัวแบ่งข้อมูล
    df = pd.read_csv(input_file, sep='\t')
    
    # เปลี่ยนชื่อคอลัมน์ให้เข้าใจง่ายขึ้น
    column_names = {
        'f1': 'f1',
        'f2': 'f2',
        'f3': 'f3',
        'f4': 'f4',
        'classlabel': 'species'
    }
    
    # เปลี่ยนชื่อคอลัมน์
    df = df.rename(columns=column_names)
    
    # แปลงค่า classlabel เป็นชื่อ species
    # species_map = {
    #     1: np.array([1.,0.,0.]),
    #     2: np.array([0.,1.,0.]), 
    #     3: np.array([0.,0.,1.])
    # }
    
    # df['species_label'] = df['species'].map(species_map)
    
    # บันทึกเป็น CSV file
    df.to_csv(output_file, index=False)
    print(f"Successfully converted {input_file} to {output_file}")


# ตัวอย่างการใช้งาน
input_file = "iris.pat"
output_file = "iris.csv"
convert_pat_to_csv(input_file, output_file)