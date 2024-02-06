# Triple Feature Propagation (TFP)

This is the code of Triple Feature Propagation (TFP) introduced in our paper: "Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding".[arxiv](https://arxiv.org/abs/2401.12798)

## Datasets

The dataset and the embedding we processed can be downloaded at [GoogleDrive](https://drive.google.com/file/d/1wptKenCyYXvIfuNXjuE2dWmbHHkib3-5/view?usp=drive_link)

* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* rel_ids_1: ids for relations in source KG;
* rel_ids_2: ids for relations in target KG;
* sup_ent_ids: training entity pairs;
* ref_ent_ids: testing entity pairs;
* triples_1: relation triples encoded by ids in source KG;
* triples_2: relation triples encoded by ids in target KG;


## Just run main.py

## Environment

* tensorflow == 2.6.0
* Python == 3.7.0
* Numpy
* tqdm

## Cite
Please consider citing this paper if you use the code or data from our work. Thanks a lot ~

```bigquery
@article{wang2024gradient,
  title={Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding},
  author={Wang, Yuanyi and Sun, Haifeng and Wang, Jingyu and Qi, Qi and Sun, Shaoling and Liao, Jianxin},
  journal={arXiv preprint arXiv:2401.12798},
  year={2024}
}
```
