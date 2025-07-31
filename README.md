# Wi-CBR
XRF55 Dataset:https://aiotgroup.github.io/XRF55/
Widar3.0 Dataset:https://tns.thss.tsinghua.edu.cn/widar3.0/

This project contains two folders, matlab and python. The former is used to preprocess and visualize Wifi data, and the later is our Wi-CBR for human behavior recognition, and please change the pretrained to 'True' of the ResNet model
If you have any question, please refer to zrb@mail.hfut.edu.cn.
## Mapping Method:

`uname-mn-ln-on-rn-rsn.dat` (rsn: 1-6)  
`envs-suname-mn-ln-on-rn.mat` (4D array in storage order: subcarrier dimension, receiver dimension, transmitter dimension, timestamp dimension)

## Mapping Table:

WIGRUNT: envs1:6750, envs2:2249, envs3:2997  

Skip empty files during loading: `12-2-2-3-5`, `13-1-1-1-1`, `13-3-3-3-5`, `14-1-1-1-1`, `15-3-1-1-5`

| Dataset File | Env Room | Action Count | User | suname ID | Data Volume |
| :----------------------------------------------------------: | :------: | :----------: | :-------------------------------------: | :--------: | :--------: |
| 20181130_user5_10_11.zip<br>20181130_user12_13_14.zip<br>20181130_user15_16_17.zip | 1 | 9 | User5,10,11<br>User12,13,14<br>User15,16,17 | 0~8 | 10125 |
| 20181205.zip | 2 | 2 (5-6) | User2 | 9 | 250 |
| 20181208.zip | 2 | 4 (1-4) | User2 | 9 | 500 |
| 20181205.zip | 2 | 3 (4-6) | User3 | 10 | 375 |
| 20181208.zip | 2 | 3 (1-3) | User3 | 10 | 375 |
| 20181209.zip | 2 | 6 | User6 | 15 | 750 |
| 20181204.zip | 2 | 9 (1-6) | User1 | 16 | 750 |
| 20181211.zip | 3 | 6 | User3,7,8,9 | 11~14 | 3000 |


**Feature set **
We will also propose the feture set we used.
