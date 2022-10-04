import pandas as pd
import numpy as np

labels_file = 'crops_defectives'
with open(f'data/{labels_file}.txt') as f:
    lines = f.readlines()


file_names, labels = [],[]
for line in lines:
    fileName, leftLabel, rightLabel = line.split('\n')[0].split(' ')

    file_names.append(f'left_{fileName}')
    file_names.append(f'right_{fileName}')

    labels.append(leftLabel)
    labels.append(rightLabel)




df = pd.DataFrame(np.array([file_names, labels]).T)
df.to_csv(f'data/{labels_file}.csv', index=False, header=False)
