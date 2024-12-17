import matplotlib.pyplot as plt
import pickle
import numpy as np
import pymatreader
import torch

from torch.utils.data import random_split

data_all = pymatreader.read_mat('/home/jagrole/AAU/9.Sem/Code/Processed_data_ALL.mat')


CGM_data = data_all['pkf']['cgm'][5]

timedata = data_all['pkf']['timecgm'][5]

doses = data_all['pkf']['doses_matched'][5]

doses_fa = data_all['pkf']['doses_matched_FA'][5]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

# Plot CGM_data over timedata in the first subplot
ax1.plot(timedata, CGM_data, label='CGM Data')
ax1.set_ylabel('CGM Data')
ax1.legend()

# Plot doses over timedata in the second subplot
ax2.plot(timedata, doses_fa, label='Doses', color='orange')
ax2.set_xlabel('Time')
ax2.set_ylabel('Doses')
ax2.legend()

# Save the figure
plt.savefig('test.png')
plt.show()
# test = np.concatenate(CGM_data)
# # test = torch.tensor(npcgm)
# splitidx = 4223934
# train = test[:splitidx]
# val = test[splitidx:]
# print(train.shape)


# plt.savefig('test.png')
# path = '/home/jagrole/AAU/9.Sem/Data/My_Vers/cgm_block2_id0'
# path = '/home/jagrole/AAU/9.Sem/cgm_block1_idx0'
# path = '/home/jagrole/AAU/9.Sem/Data/My_Vers/cgm_block1_id1'
# with open(path, 'rb') as f:
#     test = pickle.load(f)
#     plt.plot(test[1])
#     plt.savefig('test.png')
#     # print(len(test))
