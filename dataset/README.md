# WEAVESS DATASET

### ORIGIN DATASETS

|           | base_num | base_dim | query_num | query_dim | groundtruth_num/query |                           download                           |
| :-------: | :------: | :------: | :-------: | :-------: | :-------------------: | :----------------------------------------------------------: |
|  Sift1M   | 1000000  |   128    |   10000   |    128    |          100          | [sift.tar.gz](http://corpus-texmex.irisa.fr)(161MB) |
|   Gist    | 1000000  |   960    |   1000    |    960    |          100          | [gist.tar.gz](http://corpus-texmex.irisa.fr)(2.6GB) |
| Glove-100 | 1183514  |   100    |   10000   |    100    |          100          | [glove-100.tar.gz](http://downloads.zjulearning.org.cn/data/glove-100.tar.gz)(424MB) |
|   Crawl   | 1989995  |   300    |   10000   |    300    |          100          | [crawl.tar.gz](http://downloads.zjulearning.org.cn/data/crawl.tar.gz)(1.7GB) |
|   Audio   |  53387   |   192    |    200    |    192    |          20           | [audio.tar.gz](https://drive.google.com/file/d/1fJvLMXZ8_rTrnzivvOXiy_iP91vDyQhs/view)(26MB) |
|   Msong   |  992272  |   420    |    200    |    420    |          20           | [msong.tar.gz](https://drive.google.com/file/d/1UZ0T-nio8i2V8HetAx4-kt_FMK-GphHj/view)(1.4GB) |
|   Enron   |  94987   |   1369   |    200    |   1369    |          20           | [enron.tar.gz](https://drive.google.com/file/d/1TqV43kzuNYgAYXvXTKsAG1-ZKtcaYsmr/view)(51MB) |
|   UQ-V    | 1000000  |   256    |   10000   |    256    |          100          | [uqv.tar.gz](https://drive.google.com/file/d/1HIdQSKGh7cfC7TnRvrA2dnkHBNkVHGsF/view?usp=sharing)(800MB) |

### SAMPLE DATASETS (validation dataset for parameters tuning)

|           | base_num | base_dim | query_num | query_dim | groundtruth_num/query |                           download                           |
| :-------: | :------: | :------: | :-------: | :-------: | :-------------------: | :----------------------------------------------------------: |
|  Sift1M   |  10000   |   128    |    100    |    128    |          100          | [sample.sift.tar.gz](https://drive.google.com/file/d/1ItZpZHn8ALBG4th3ede6O_xUbXHyP2uM/view?usp=sharing)(1.5MB) |
|   Gist    |  10000   |   960    |    100    |    960    |          100          | [sample.gist.tar.gz](https://drive.google.com/file/d/15Na0AmmHxX7HnckogKYb8MvJz5WXNoVZ/view?usp=sharing)(18MB) |
| Glove-100 |  10000   |   100    |    100    |    100    |          100          | [sample.glove-100.tar.gz](https://drive.google.com/file/d/18cnlaKxrkq3NFGP9O6OzpdISUtVoNZoY/view?usp=sharing)(3.6MB) |
|   Crawl   |  10000   |   300    |    100    |    300    |          100          | [sample.crawl.tar.gz](https://drive.google.com/file/d/12x-HuNJA6BCFXyxKSQiUAbqZh8Alwr-d/view?usp=sharing)(8.4MB) |
|   Audio   |  10000   |   192    |    100    |    192    |          100          | [sample.audio.tar.gz](https://drive.google.com/file/d/1UDjUocqOVVHnK--WYRPygACUY8zSXeyF/view?usp=sharing)(4.8MB) |
|   Msong   |  10000   |   420    |    100    |    420    |          100          | [sample.msong.tar.gz](https://drive.google.com/file/d/1wTIvK3aDz2cOcO30jlsNNRbNiPtWJBk-/view?usp=sharing)(14MB) |
|   Enron   |  10000   |   1369   |    100    |   1369    |          100          | [sample.enron.tar.gz](https://drive.google.com/file/d/1LLjcjy8C1ylUtlceNCxPVvhbMPf6fyGr/view?usp=sharing)(5.5MB) |
|   UQ-V    |  10000   |   256    |    100    |    256    |          100          | [sample.uqv.tar.gz](https://drive.google.com/file/d/1WXU0JAYSffsay_J4j___UzaLwCkm19yq/view?usp=sharing)(8.2MB) |

### SYNTHETIC DATASETS

In following table, `base_cluster` indicates the number of clusters on base or query data, `deviation` indicates the standard deviation of the distribution in each cluster. For a more detailed description about synthetic datasets, please refer to our paper.

|           | base_num | base/query_dim | base_cluster | deviation | groundtruth_num/query | download |
| :-------: | :------: | :------: | :-------: | :-------: | :-------------: | :-------------: |
| d_8    | 100000    | 8      | 10  | 5       | 100            | [d_8.tar.gz](https://drive.google.com/file/d/1bG-dCDeYDgpF7EriNpmM3Wh3NccHTBTU/view?usp=sharing)(3.3MB) |
| d_32      | 100000    | 32      | 10      | 5       | 100            | [d_32.tar.gz](https://drive.google.com/file/d/1tDMgCIXoSmPdu4SO6L9D3xYOiZvZ-tB_/view?usp=sharing)(12MB) |
| d_128 | 100000    | 128      | 10      | 5       | 100            | [d_128.tar.gz](https://drive.google.com/file/d/1B2TTgaWJdNg0-fO27zgzekrynrQIpryn/view?usp=sharing)(117MB) |
| n_10000     | 10000    | 32      | 10       | 5       | 100             | [n_10000.tar.gz](https://drive.google.com/file/d/1cnoP5RAHxUrJ4oykRhzyGAiSXY4N-hUu/view?usp=sharing)(1.2MB) |
| n_100000     | 100000    | 32      | 10      | 5       | 100           | [n_100000.tar.gz](https://drive.google.com/file/d/1U2t_uw0nPTm1W8ZDlTiIfEZp4MeYHDyv/view?usp=sharing)(12MB) |
| n_1000000     | 1000000    | 32      | 10     | 5       | 100            | [n_1000000.tar.gz](https://drive.google.com/file/d/1JHXo-8AXskcpyXJLAyrDmusCGigboVgq/view?usp=sharing)(120MB) |
| c_1     | 100000    | 32     | 1    | 5      | 100            | [c_1.tar.gz](https://drive.google.com/file/d/1RMD5zeQo-ZcvP6f-XY6UdlU1NhlZzLhO/view?usp=sharing)(12MB) |
| c_10      | 100000    | 32      | 10   | 5       | 100            | [c_10.tar.gz](https://drive.google.com/file/d/1aoDSfSJ--51gLfPBS5I3m6RU_PzlokVF/view?usp=sharing)(12MB) |
| c_100      | 100000    | 32      | 100  | 5       | 100            | [c_100.tar.gz](https://drive.google.com/file/d/11aEPf8Fq6Q3P1V8UT9MJ1FlThpNE3J7v/view?usp=sharing)(12MB) |
| s_1      | 100000    | 32      | 10   | 1      | 100            | [s_1.tar.gz](https://drive.google.com/file/d/1lSZtRkqlxpem_uORxGo48_JvN9SF-S5h/view?usp=sharing)(12MB) |
| s_5      | 100000    | 32      | 10   | 5       | 100            | [s_5.tar.gz](https://drive.google.com/file/d/1L1OySfOaY27U3__If2FfMvRYfaaov3JB/view?usp=sharing)(12MB) |
| s_10      | 100000    | 32      | 10 | 10     | 100            | [s_10.tar.gz](https://drive.google.com/file/d/1KeTeSlVfzlltLwfSfFp7KSSSl-ksY2S7/view?usp=sharing)(12MB) |

