import pandas as pd
import numpy as np

dt = pd.read_csv("car.data")
# số hàng có trong data set
amount_of_rows = len(dt)
# lấy giá trị của header cột
header = dt.columns.values.tolist()
# thuộc tính
attributes = header[0:-1]

label_name = header[-1]

# lấy giá trị của nhãn
labels = np.unique(np.array(dt[label_name]))

# tạo một dictionary rỗng để lưu các giá trị của thuộc tính và số lần xuất hiện của nó trong bảng giá trị của thuộc tính
attributes_infor = {}
# tạo dictionary rỗng để lưu thông tin các label
labels_infor = {}

# đếm số lượng của mỗi nhãn
for label in labels:
    labels_infor[label] = (np.array(dt[label_name])).tolist().count(label)


for attribute in attributes:
    # các giá trị của thuộc tính (attribute)
    values_of_attr = np.unique(np.array(dt[attribute]))
    # tạo một dictionary rỗng để lưu giá trị của thuộc tính và số lượng của nó
    attributes_infor[attribute] = {}
    for value in values_of_attr:
        # tính số lượng của giá trị(valư) xuất hiện trong thuộc tính (attribute)
        attributes_infor[attribute][value] = np.array(dt[attribute]).tolist().count(value)

print(attributes)

print(labels)

for label in labels:
    print(label + ': ' + str(labels_infor[label]) + ' (' + str(round(labels_infor[label] / amount_of_rows ,5) * 100 ) + ').')

# hàm in thông tin thuộc tính
def print_attribute_infor():
    for attribute in attributes:
        print(attribute + ':')
        for value in attributes_infor[attribute]:
            print('     ' + value +': ' + str(attributes_infor[attribute][value]) + ' (' + str(round(attributes_infor[attribute][value]/amount_of_rows, 5) * 100) + '%).')


print_attribute_infor()