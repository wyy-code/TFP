# Triple Feature Propagation (TFP)

This is the code of Triple Feature Propagation (TFP) introduced in our paper: "Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding".

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

## Cite
Please consider citing this paper if you use the code or data from our work. Thanks a lot ~

```bigquery
@misc{wang2024gradient,
      title={Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding}, 
      author={Yuanyi Wang and Haifeng Sun and Jingyu Wang and Qi Qi and Shaoling Sun and Jianxin Liao},
      year={2024},
      eprint={2401.12798},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
